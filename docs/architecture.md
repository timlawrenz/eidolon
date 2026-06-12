# Eidolon — Architecture

> **⚠️ This document has been updated to reflect the post-Phase-4 empirical record
> (2026-06-11).** The v1 design envisioned a 3-partition E vector with decoupled
> geometry/depth/normals. Phases 2–4 falsified two of three partitions. The
> architecture described below is the *empirically validated* design — what
> survived the gate, not what was originally planned. See
> `02_EXPERIMENTS_AND_RESULTS.md` for the full experimental record.

## 0. North Star

**Eidolon describes a face — its shape and identity — in a conditioning stack.**

The deliverable is an interpretable geometry vector `z_g` plus an identity
conditioning stream from DINOv3. Internally, the geometry partition is
*mathematically firewalled* so the disentanglement built into the encoder
survives all the way into the DiT's attention mechanism.

```
E_structured = [ z_g ]   ∈ ℝ^50  (interpretable geometry control)
                  │
                  └────────── geometry / shape  (z_g from facial keypoints, pose-invariant)

Identity stream: flesh-masked DINOv3 patch tokens  (external to E_structured, AUC 0.797)
```

**Former partitions — both conclusively dead (see ledger Phases 2 & 2b):**
- `z_d` (depth): ΔAUC −0.023 to −0.034 — monocular relative depth dilutes identity signal.
- `z_a` (normals/surface): ΔAUC −0.039 — initial PASS overturned; normals hallucinate generic geometry.
- **Theory:** monocular volumetric models produce plausible-but-generic human geometry; they do not encode identity-specific micro-curvature.

The conditioning stack is now **2-stream**: DINO patches (identity) + z_g (control).

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

#### Shipped production encoder (as-built — see ledger Phase 1-R FINAL)

The frozen `z_g` encoder that actually shipped:

- **Fit corpus:** 69,851 FFHQ faces from `stratum-ffhq`. **k = 50**, retained
  variance **99.987%**, full fit ~107s.
- **Pipeline (exact order):** `pose.npy` → slice indices 23–90 (68 iBUG pts) →
  drop confidence channel → **mean-confidence prefilter** (drop a face if its
  *mean* DWPose confidence < 0.5) → **3D frontalize against the canonical 300W
  template at z_scale = 1.0** → light 2D GPA (center/scale/roll) → PCA → whiten.
- **z_scale = 1.0 rationale:** uses the 300W template's anatomical depth at face
  value; captures ~85% of the achievable within-identity-scatter (S_W) reduction
  without extrapolating depth beyond anatomy. (Sweep argmax was z=2.0 for a
  negligible J gain; 1.0 was chosen on anatomical grounds — see ledger.)
- **Frozen artifact:** `experiments/geometry_pca/output/encoder_production.npz`.
  Contents: `components` (50,136), `canonical_template` (68,3), `pca_mean` (136,),
  `whiten_mu` (50,), `whiten_sigma` (50,), `gpa_mean` (68,2), plus
  `explained_variance_ratio` (50,). The canonical template is persisted *inside*
  the encoder so inference frontalization is reproducible.

> **Storage rule (project convention):** caching CPU/GPU labor is encouraged
> (persist decoded `.npy`, depth caches, frozen encoders), but generated
> artifacts **must not live on local disk** — they go to the NAS project folder.
> `experiments/geometry_pca/data/` is a symlink to
> `/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/`; write all caches
> under `data/`. fscache accelerates reads; eat the slower NAS pass otherwise.

### 3.4 Validation gate (go / no-go)

1. **Scree / cumulative-variance curve** — confirm ~99% variance retained at K≈50.
2. **Reconstruction error vs. K** — quantify fidelity at the chosen K.
3. **±3σ traversal visualization** — scatter-plot traversals of `C_1…C_5` and
   eyeball that they correspond to real morphology (jaw width, eye distance) and
   **not residual camera pose**. If `C_1` is still global rotation, **GPA failed
   and is fixed here, cheaply**, before any downstream work.

---

## 4. The Volumetric Encoders (`z_d`, `z_a`) — **BOTH DEAD**

Both volumetric partitions were built and conclusively killed by the verification-AUC
gate (see ledger Phases 2 & 2b). The identical fit-PCA-store-whitening recipe was
applied to Sapiens maps, but neither depth nor surface normals carry complementary
identity signal beyond 2D facial geometry:

- `z_d` (depth): from `depth.npy`. **(DEAD — ΔAUC −0.023 to −0.034).**
  Confirmed at 24× facial resolution with domain shift eliminated. Monocular relative
  depth is fundamentally entangled with camera distance; it dilutes identity signal.
- `z_a` (surface normals): from `normal.npy`, masked by `seg.npy`. **(DEAD — ΔAUC −0.039).**
  Initial PASS (2026-06-10) was overturned on face-crop re-run; the PASS was an artifact
  of the low editorial-keypoint z_g baseline (0.540 → 0.688). At proper resolution,
  normals *subtract* identity signal. The canonical variant was `xy` (nz dropped as
  redundant). PCA extracts 50 components natively.

**Scientific conclusion:** Monocular volumetric models (Sapiens) hallucinate
plausible, topologically accurate, but *generic* human geometry. They do not encode
the identity-specific biological micro-curvature required for face recognition.
The "fast path" — deriving decoupled structural sliders from monocular networks —
is a definitive dead end.

**Implementation wrinkle:** normals are dense `H×W` maps, so the data matrix
will not fit in RAM. Decoded normal caches are written under the NAS `data/`
symlink (see storage rule in §3.3), e.g. `data/normal_cache/ffhq_normal_raw.npy`,
from which the `xy` variant is derived in RAM during fitting.

**Validation gate for partitions (pre-registered):** beyond the §3.4 scree checks,
any partition must earn its place in `E` via an **incremental-information** test
on the reviewed hegre identity corpus (`data/review.db`, clean identities only):

> `AUC([z_g | z_x]) > AUC(z_g) + ε`  (Verification AUC)

i.e. concatenating the partition onto geometry must lift same/different identity
discrimination above the measured seed-noise floor (ε=0.01). Note: the old
trace-Fisher J test was deprecated (it is a weighted average blind to complementarity).

> ⚠️ **PERMANENT WARNING — the 2.5D rotation trap (visibility bias).**
> Never apply a full 3D de-rotation (Rᵀ) to *partially observed* 2.5D surface
> data (normals, depth gradients, any camera-hemisphere field). The camera only
> sees the forward-facing hemisphere, so the field's **global mean is
> camera-locked** (pose-blind). De-rotating pointwise rotates that camera-locked
> mean by Rᵀ — the mean then **traces the head trajectory exactly**, injecting
> pose into the centroid (measured: corr(mean nₓ, yaw) −0.008 raw → **−0.935**
> de-rotated). PCA promotes that swinging centroid into C1 and recreates the
> double-conditioning conflict with the DiT's `pose.npy` sequence that
> Phase 1-R eliminated. Pose-normalize 2.5D fields some other way (e.g. drop
> deterministic channels, canonical *crops*), never by global vector rotation.

> **[OVERTURNED 2026-06-11] Macro architectural finding (Phase 2b, original):**
> The claim that `z_a` beats `z_g` (0.562 vs 0.540) and that "the surface partition
> carries the biological identity" was invalidated when the z_g baseline was corrected
> from the editorial-keypoint-resolution artifact (0.540) to the proper face-crop
> measurement (0.688). Normals alone (0.587) do not beat corrected z_g (0.688).
> The "identity-dominant partition" framing is rescinded. See ledger Phase 2b
> face-crop overturn.

---

## 5. The DINOv3 Bridge (premise validation) — **DEAD**

Linear-regress `dinov3_cls` (1024-d) → the whitened geometric components.
This was attempted as both a premise validation and a "fast path" to map DINO
embeddings to interpretable physical sliders for DiT conditioning.

**Result (Phase 3): both directions are dead.**
- **R² premise:** z_g R² = 0.690 (FAIL on C1–C10 ≥ 0.6 criterion), z_a R² = 0.385 (FAIL).
  DINO cannot faithfully reconstruct fine surface curvature.
- **Identity transfer:** the bridge Ŷ_g AUC (0.704) is *no better than random 50-d
  projections of its own input* (0.712 ± 0.007). Linear regression destroys identity
  when forced to map to physical geometry.

**What survived:** raw `dinov3_cls` on face crops is the strongest identity carrier
measured (AUC 0.766), and Phase 4 improved this further with masked patch pooling
(AUC 0.797). DINO carries identity — it just cannot reconstruct our sliders.

**Lesson:** every transfer gate must include a random-projection null of its input
representation. A gate without the right null can "pass" on structure it never isolates.

> **[RESCINDED]** The R² was originally designated as "the single most important number
> in the project" — the project go/no-go signal. The bridge failed; the project did not
> stop, because (a) raw DINO proved itself a stronger identity carrier than any of our
> structured encoders, and (b) E's decoupled control remains non-redundant (DINO cannot
> reconstruct it). The go/no-go framing was too narrow.

---

## 6. Scoring (inference)

**Geometry control:** To score any image for z_g: project its preprocessed,
GPA-aligned, flattened vector onto the frozen PCA components, then whiten.
Output is a vector of absolute scalar Z-scores — the exact slider positions for
every attribute, computed in milliseconds, zero training compute.

**Identity:** For identity conditioning, extract flesh-masked DINOv3 patch tokens
from the face crop (see Phase 4, `37_dino_patch_face_pooling.py`). The pooled
1×1024 vector is the compact fallback; for DiT conditioning, prefer the unpooled
masked patch tokens (~100–1,900 face tokens) via cross-attention.

---

## 7. DiT Fusion Architecture (DEFINITIVE)

The geometry encoder produces a 50-d vector `z_g`. Identity comes from DINO
patch tokens via a parallel cross-attention stream. The danger is naively
*re-coupling* the modalities right before attention. The solution is a
**Strategy-C-structure / Strategy-B-mechanics hybrid**: one vector on the
outside, a mathematical firewall on the inside.

### 7.1 Block-diagonal ingestion

The conditioning MLP that ingests `z_g` **must be block-diagonal** relative to
any future structured partitions. A *dense* projection would re-entangle
partitions on the very first matmul, destroying the orthogonality the encoders
worked to produce.

**Current state:** `n_parts=1` (z_g only). Identity comes from DINO patches via
a parallel cross-attention stream. The block-diagonal architecture is designed to
accommodate future structured partitions without re-entanglement — if a new
partition survives the verification-AUC gate, it slots into a new expert module
without modifying the existing ones.

#### Implementation rule — independent modules, NEVER a masked Linear

> **Do NOT** implement the firewall as a single masked `nn.Linear`. This is the
> easiest way to *silently* compromise the firewall during training.

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

**The structurally sound approach: strictly independent modules.** Make the
block-diagonal structure a property of the computation graph itself — there is no
off-diagonal weight to leak through, because those parameters *do not exist*.

```python
class BlockDiagonalIngestion(nn.Module):
    """Strict modality firewall: independent per-partition MLPs, no shared weights.
    Enforced by module topology, not by masking — no off-diagonal parameter exists,
    so nothing can drift through optimizer state or weight decay."""
    def __init__(self, part_dim: int = 50, hidden_dim: int = 256, n_parts: int = 1):
        super().__init__()
        # ONE module per partition. Currently n_parts=1 (z_g only).
        # New surviving partitions add experts without modifying existing ones.
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(part_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_parts)
        )
        self.part_dim = part_dim

    def forward(self, z_g: torch.Tensor) -> list[torch.Tensor]:
        parts = z_g.split(self.part_dim, dim=-1)     # (B,50) → 1×(B,50)
        return [expert(p) for expert, p in zip(self.experts, parts)]
```

The geometry expert `mlp_g` sees only its own 50 whitened scalars; the isolated
stream then feeds the expanded-token step (§7.3) and the decoupled cross-attention
head (§7.2). Identity comes from DINO patch tokens via a parallel attention stream.
The firewall holds by construction — no per-step mask babysitting, no
optimizer-state leakage.

### 7.2 Decoupled / parallel cross-attention (IP-Adapter style)

**Two-stream design:** geometry gets its own K/V projections and cross-attention
sub-layer; DINO identity patches get their own. Contributions are **summed**,
never forced to share a softmax budget:

```
h' = h + λ_g · Attn_g(h, z_g_tokens) + λ_dino · Attn_dino(h, dino_patches)
```

- **Why:** softmax attention is a competition — weights over a merged sequence
  must sum to 1, so a high-variance geometry token can suppress a subtle identity
  token. Independent K/V + summation lets a patch attend *fully* to geometry AND
  *fully* to identity simultaneously.
- The `λ` scalars double as **inference-time strength sliders**
  (turn geometry control up/down, identity up/down).

### 7.3 Positional encoding — EXPANDED TOKENS (not a fat token)

Do **not** map the 50-d partition into a single "fat" K/V token — that forces the
DiT into an all-or-nothing decision per patch ("attend to geometry, or don't").

Instead, expand each scalar component `s_i` into its **own token** via a learned
per-component embedding `v_i ∈ ℝ^D`:

```
Seq_g = [ s_1·v_1, s_2·v_2, … , s_50·v_50 ]
```

This turns the scalar array into an **unordered set of 50 tokens**, so
cross-attention can compute 50 separate attention weights per pixel patch
(query eye-distance heavily, ignore jaw-width — for a given patch). ~50 extra
tokens is trivial for modern DiT context windows.

### 7.4 Whitening — MANDATORY

Whiten every PCA component to zero-mean / unit-variance **before** it touches the
network:

```
s'_i = (s_i − μ_i) / σ_i
```

PCA variance decays exponentially — unwhitened, top components' gradients
dominate the optimizer and micro-features vanish. Whitened: every slider sits on
equal footing, and the network learns its **own** internal scaling weights based
purely on loss, free of the numerical bias of the original 2D pixel coordinates.

### 7.5 Final flow

```
Structured control:  single dense vector  z_g ∈ ℝ^50
     │
Ingestion:           block-diagonal MLP (1 expert, n_parts=1)
     │
Expansion:           each scalar × its learned embedding  →  50 geometry tokens
     │
Attention:           decoupled cross-attention — geometry queried independently,
                     DINO identity patches queried in parallel, SUMMED (λ sliders)
```

---

## 8. Build Order (dependency chain)

The encoder (§3) is **deterministic linear algebra** — built and fully validated
without training anything. Phases 1–4 are complete; Phase 5 is the next build
target.

| Phase | Scope | Validation gate | Status |
|-------|-------|-----------------|--------|
| **1/1-R** | Geometry encoder `z_g` (§3) | scree, recon error, ±3σ traversal, pose-invariance probe | ✅ CONCLUDED — PASS |
| **2** | Depth encoder `z_d` (§4) | verification AUC gate | ❌ CONCLUDED — FAIL |
| **2b** | Surface normals encoder `z_a` (§4) | verification AUC gate | ❌ CONCLUDED — FAIL |
| **3** | DINOv3 bridge (§5) | R² premise + identity transfer | ❌ CONCLUDED — FAIL |
| **4** | Masked DINO patch tokens | AUC gate + cross-shoot verification | ✅ CONCLUDED — PASS |
| **5** | DiT conditioning stack (§7) | training convergence | 🔜 TBD |

**Phase 5 is the next build target.** Architecture is settled: 2-stream decoupled
cross-attention (DINO patches for identity + z_g expanded tokens for interpretable
geometry control), block-diagonal ingestion, whitened inputs.

---

## Appendix — Ecosystem

Related repos / assets (timlawrenz):

- `github.com/timlawrenz/stratum-hq`
- `github.com/timlawrenz/prx-tg`
- `github.com/timlawrenz/morphometrics`
- `github.com/timlawrenz/image_embed`
- `github.com/timlawrenz/eidolon`
- `huggingface.co/datasets/timlawrenz/stratum-ffhq`
