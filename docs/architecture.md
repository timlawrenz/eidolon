# Eidolon — Architecture

## 0. North Star

**Eidolon describes a person — face, albedo, and body — in a single vector.**

The deliverable is one concatenated embedding `E` that can be stored on disk,
passed across an API as a flat array, and used to condition a generative model.
Internally, however, `E` is rigorously partitioned and the partitions are
*mathematically firewalled* from one another so that the disentanglement we build
into the encoder survives all the way into the model's attention mechanism.

```
E = [ z_g | z_d | z_a ]   ∈ ℝ^(50 + 50 + 50)
      │     │     │
      │     │     └── albedo / surface  (from normal maps)
      │     └──────── depth / volume    (from depth maps)
      └────────────── geometry / shape  (from facial keypoints)
```

The component counts (≈50 each) are targets, finalized per-modality by the
variance-retention validation gate (see §3.4 and §6).

---

## 1. Data Source

Full local copy of the `stratum-ffhq` dataset:

```
/mnt/nas-ai-models/training-data/ffhq/stratum/
```

### Layer formats

| Layer    | Storage                | Per-image contents |
|----------|------------------------|--------------------|
| caption  | main data parquet (`data/`) | `image_id`, `width`, `height`, `aspect_bucket`, `caption` |
| dinov3   | tar archives           | `dinov3_cls.npy` (1024, f16), `dinov3_patches.npy` (N×1024, f16) |
| t5       | tar archives           | `t5_hidden.npy` (512×1024, f16), `t5_mask.npy` (512, u8) |
| pose     | tar archives           | `pose.npy` (133×3, f16) — COCO-WholeBody keypoints in [-1, 1] |
| seg      | tar archives           | `seg.npy` (H×W, u8) — 28-class body-part segmentation (Sapiens) |
| depth    | tar archives           | `depth.npy` (H×W, f16) — relative depth, foreground-masked (Sapiens) |
| normal   | tar archives           | `normal.npy` (H×W×3, f16) — unit surface normals, foreground-masked (Sapiens) |

---

## 2. Core Principle — Orthogonality via PCA

Each modality's sub-vector is produced by **PCA on aligned, denoised data**.
PCA mathematically guarantees that every component is strictly orthogonal to
every other component. Unlike entangled latent spaces (DINOv3, CLIP), this gives
us **perfectly isolated sliders**: changing component `C_14` (e.g. eye distance)
has zero linear correlation with `C_5` (e.g. jaw length).

This is, in effect, a bespoke **Statistical Shape Model (SSM)** — the geometric
half of a 3D Morphable Model — derived entirely from 2D predictive layers.

---

## 3. The Geometry Encoder (`z_g`)

### 3.1 Keypoint slicing

`pose.npy` is the 133-point **COCO-WholeBody** layout, structured sequentially:

| Indices | Region | Action |
|---------|--------|--------|
| 0–16    | Body joints   | **Discard** |
| 17–22   | Foot keypoints | **Discard** |
| **23–90** | **Facial landmarks (68 pts)** | **KEEP** |
| 91–132  | Hand keypoints | **Discard** |

Indices **23–90** are 68 dense facial keypoints matching the industry-standard
**300W / iBUG** landmark configuration (stable anchors for jawline, eyebrows,
nose bridge, lip contours).

Drop the 3rd channel (**confidence**) — it is an artifact of the DWPose model's
uncertainty, not a geometric truth. Keep only `(x, y)`:

```
(133, 3)  →  slice [23:91]  →  (68, 3)  →  drop conf  →  (68, 2)  →  flatten  →  ℝ^136
```

### 3.2 Alignment — pose-invariance is MANDATORY (by construction)

> **North-Star constraint (added 2026-06-09, after Phase 1 was reopened):**
> `z_g` MUST be mathematically orthogonal to head pose. Identity is *invariant*;
> pose is *transient*. A vector that encodes transient camera state is
> disqualified from being an identity descriptor. This is not a cleanup
> preference — it is a definitional requirement of `E`.

#### Why this is non-negotiable (the redundancy / entanglement trap)

1. **Semantic category error.** The North Star (§0) says `E` describes the
   *invariant person*. Yaw/pitch are exactly the things that change when the
   identity does not. Encoding them in `E` makes `E` partly a description of a
   camera event.
2. **Double-conditioning conflict.** The DiT already ingests the raw `pose.npy`
   stream as its authoritative spatial-orientation signal. If `z_g` *also*
   encodes head orientation, the same physical fact is conditioned down two
   paths that can disagree — a direct optimization conflict (a high-C₁ `z_g`
   would fight the spatial coordinates on the primary pose path). `z_g` must be
   the **pose-orthogonal complement** of what `pose.npy` already carries.

#### Why plain 2D GPA is insufficient

2D GPA only neutralizes **2D** transforms — translation, uniform scale, and
in-plane rotation (roll). It is **mathematically blind to out-of-plane 3D
rotation (yaw and pitch)**. A head turning left↔right produces huge variance in
2D coordinate space, so PCA — doing exactly what it is optimized to do — shoves
that variance straight into the top components. Empirically (Phase 1), this is
exactly what happened: C₁ = yaw, C₂ = pitch. The math was correct; the objective
was wrong.

#### Mandated approach: 3D-aware canonical alignment

Recover a canonical, frontal 3D frame **before** PCA so pose is factored out by
construction rather than hoped-into-discardable-components:

1. Estimate per-sample head rotation from the 68 keypoints (EPnP / PnP against a
   canonical 3D mean-face template; escalate to a full 3DMM fit only if the
   lightweight estimate proves insufficient — see ledger Phase 1-R).
2. Rotate the points back to a rigid `(0,0,0)` frontal orientation in 3D.
3. Reproject to 2D and run the existing center+scale+roll GPA on the now
   pose-normalized shapes.
4. Fit PCA on these. The remaining variance is **pose-invariant biological
   morphology**.

**Depth bonus:** rotating a profile to frontal does NOT discard the profile-only
signal (nose projection, brow ridge, chin protrusion). It *preserves* it — the
reprojected X-spread of e.g. the nose keypoints now encodes the depth of that
projection. This is strictly superior to a frontal-only data filter, which would
delete those identity-bearing views outright.

#### New validation gate (supersedes the Phase 1 gate)

Beyond scree + traversal, add a **pose-invariance probe**: take one identity,
synthesize several yaw/pitch variants of its keypoints, encode each, and assert
the resulting `z_g` vectors are near-identical (low variance across the
synthetic-pose set). C₁ must now read as morphology, not orientation.

### 3.3 PCA fit

Stack all `N` aligned, flattened faces into `M ∈ ℝ^(N × 136)`. Fit PCA, keep the
top-K components. **Persist the components AND the whitening statistics
(μ_i, σ_i) as the frozen encoder.**

### 3.4 Validation gate (go / no-go)

1. **Scree / cumulative-variance curve** — confirm ~99% variance retained at K≈50.
2. **Reconstruction error vs. K** — quantify fidelity at the chosen K.
3. **±3σ traversal visualization** — scatter-plot traversals of `C_1…C_5` and
   eyeball that they correspond to real morphology (jaw width, eye distance) and
   **not residual camera pose**. If `C_1` is still global rotation, **GPA failed
   and is fixed here, cheaply**, before any downstream work.

---

## 4. The Volumetric Encoders (`z_d`, `z_a`)

Apply the **identical** fit-PCA-store-whitening recipe to the Sapiens maps:

- `z_d` (depth / volume): from `depth.npy`, background removed via `seg.npy` mask.
- `z_a` (albedo / surface): from `normal.npy`, background removed via `seg.npy` mask.

These give independent, mathematically decoupled sliders for facial volume and
surface/lighting — letting us deepen a cheekbone shadow without shifting the
jawline.

**Implementation wrinkle:** depth/normal are dense `H×W` maps, so the data matrix
will not fit in RAM. Use **incremental / randomized SVD** rather than a full
in-memory PCA. Same validation gate as §3.4 (scree plot + traversal viz).

---

## 5. The DINOv3 Bridge (premise validation)

Linear-regress `dinov3_cls` (1024-d) → the whitened geometric components.

> **The R² of this regression is the single most important number in the project.**
> It measures how much of our interpretable physical geometry actually lives
> *linearly* in the DINO latent space. High R² → the semantic embeddings genuinely
> encode our sliders. Low R² → geometry and semantics are more decoupled than hoped.

This also yields a fast path to map abstract AI embeddings → interpretable
physical sliders for DiT conditioning context.

---

## 6. Scoring (inference)

To score any image: project its preprocessed, GPA-aligned, flattened vector onto
the frozen PCA components, then whiten. Output is a vector of absolute scalar
Z-scores — the exact slider positions for every attribute, computed in
milliseconds, zero training compute.

---

## 7. DiT Fusion Architecture (DEFINITIVE)

The encoders above produce a single concatenated vector `E`. The danger is
naively *re-coupling* the modalities right before attention. The solution is a
**Strategy-C-structure / Strategy-B-mechanics hybrid**: one vector on the
outside, a mathematical firewall on the inside.

### 7.1 Block-diagonal ingestion

The conditioning MLP that ingests `E` **must be block-diagonal**. A *dense*
projection (`nn.Linear(150, hidden)`) would re-entangle the partitions on the
very first matmul, destroying the orthogonality the encoders worked to produce.

#### Implementation rule — three independent modules, NEVER a masked Linear

> **Do NOT** implement the firewall as a single `nn.Linear(150, hidden_dim)`
> with its off-diagonal weights masked to zero. This is the easiest way to
> *silently* compromise the firewall during training.

**Why masking leaks (and re-entangles over thousands of steps):**

1. **Optimizer state, not just gradients.** Even when the gradient through a
   zeroed weight is zero, optimizers like Adam/AdamW carry per-parameter
   momentum (`exp_avg`) and second-moment (`exp_avg_sq`) buffers. A masked-zero
   weight with a nonzero buffer gets nudged off zero by the update
   `m̂ / (√v̂ + ε)`. Unless the mask is re-applied **after every
   `optimizer.step()`**, the off-diagonal weights drift nonzero — and a single
   nonzero off-diagonal weight is a live wire between two partitions.
2. **Weight decay.** AdamW's decoupled decay touches every parameter regardless
   of gradient and does not restore a drifted weight to *exactly* zero on the
   mask's schedule. The zeros rot.

A masked Linear is therefore not a structural guarantee — it is a constraint you
must babysit every step, and one missed re-application permanently re-entangles
the modalities. The failure is silent, gradual, and only visible thousands of
steps later as degraded disentanglement.

**The structurally sound approach: three strictly independent modules.** Make the
block-diagonal structure a property of the computation graph itself — there is no
off-diagonal weight to leak through, because those parameters *do not exist*.

```python
class BlockDiagonalIngestion(nn.Module):
    """Strict modality firewall: independent per-partition MLPs, no shared weights.
    Enforced by module topology, not by masking — no off-diagonal parameter exists,
    so nothing can drift through optimizer state or weight decay."""
    def __init__(self, part_dim: int = 50, hidden_dim: int = 256, n_parts: int = 3):
        super().__init__()
        # ONE module per partition (mlp_g / mlp_d / mlp_a). No nn.Linear(150, ...).
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(part_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_parts)
        )
        self.part_dim = part_dim

    def forward(self, E: torch.Tensor) -> list[torch.Tensor]:
        parts = E.split(self.part_dim, dim=-1)          # (B,150) → 3×(B,50)
        return [expert(p) for expert, p in zip(self.experts, parts)]
```

Each `mlp_g / mlp_d / mlp_a` sees only its own 50 whitened scalars; the three
isolated streams then feed the expanded-token step (§7.3) and the decoupled
cross-attention heads (§7.2). The firewall holds by construction — no per-step
mask babysitting, no optimizer-state leakage.

### 7.2 Decoupled / parallel cross-attention (IP-Adapter style)

Each modality gets its **own** K/V projection matrices and its **own**
cross-attention sub-layer. Contributions are **summed**, never forced to share a
softmax budget:

```
h' = h + λ_g · Attn_g(h, z_g) + λ_d · Attn_d(h, z_d) + λ_a · Attn_a(h, z_a)
```

- **Why:** softmax attention is a competition — weights over a merged sequence
  must sum to 1, so a high-variance geometry token can suppress a subtle albedo
  token. Independent K/V + summation lets a patch attend *fully* to geometry AND
  *fully* to albedo simultaneously.
- The `λ` scalars double as **per-modality inference-time strength sliders**
  (turn geometry up, lighting down).

### 7.3 Positional encoding — EXPANDED TOKENS (not a fat token)

Do **not** map a 50-d partition into a single "fat" K/V token — that forces the
DiT into an all-or-nothing decision per patch ("attend to geometry, or don't").

Instead, expand each scalar component `s_i` into its **own token** via a learned
per-component embedding `v_i ∈ ℝ^D`:

```
Seq_g = [ s_1·v_1, s_2·v_2, … , s_50·v_50 ]
```

This turns the scalar array into an **unordered set of 50 tokens**, so
cross-attention can compute 50 separate attention weights per pixel patch
(query eye-distance heavily, ignore jaw-width — for a given patch). ~150 extra
tokens total is trivial for modern DiT context windows.

### 7.4 Whitening — MANDATORY

Whiten every PCA component to zero-mean / unit-variance **before** it touches the
network:

```
s'_i = (s_i − μ_i) / σ_i
```

PCA variance decays exponentially (`C_1` raw variance ≈ 1000, `C_50` ≈ 0.01).
Unwhitened: `C_1` gradients dominate the optimizer and `C_50` (micro-features)
vanish — the network ignores them. Whitened: every slider sits on equal footing,
and the network learns its **own** internal scaling weights based purely on loss,
free of the numerical bias of the original 2D pixel coordinates.

### 7.5 Final flow

```
Disk / API:   single dense vector  E ∈ ℝ^150
     │
Ingestion:    slice → 3 whitened partitions  [50] [50] [50]
     │
Expansion:    each scalar × its learned embedding  →  3 sequences of 50 tokens
     │
Attention:    decoupled cross-attention — geometry / depth / albedo queried
              independently and SUMMED (block-diagonal projections, λ sliders)
```

---

## 8. Build Order (dependency chain)

The encoder (§3–§6) is **deterministic linear algebra** — buildable and fully
validatable *without training anything*. If the encoder is wrong, no clever
cross-attention saves us, so we de-risk it first.

| Phase | Scope | Validation gate |
|-------|-------|-----------------|
| **1** | Geometry encoder `z_g` (§3) | scree curve, recon error, ±3σ traversal viz |
| **2** | Volumetric encoders `z_d`, `z_a` (§4) | same recipe; randomized SVD for dense maps |
| **3** | DINOv3 bridge (§5) | regression **R²** = project go/no-go signal |
| **4** | DiT conditioning stack (§7) | training convergence |

**Phase 1 is the first build target.** It produces a working, inspectable `z_g`
artifact and proves the geometric-slider concept end-to-end before any
volumetric or DiT work is committed.

---

## Appendix — Ecosystem

Related repos / assets (timlawrenz):

- `github.com/timlawrenz/stratum-hq`
- `github.com/timlawrenz/prx-tg`
- `github.com/timlawrenz/morphometrics`
- `github.com/timlawrenz/image_embed`
- `github.com/timlawrenz/eidolon`
- `huggingface.co/datasets/timlawrenz/stratum-ffhq`
