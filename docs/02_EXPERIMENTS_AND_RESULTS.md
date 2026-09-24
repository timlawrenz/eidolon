# Experiments & Results

This ledger documents the empirical findings and definitive conclusions of all experiments in the Eidolon repository. Negative results are recorded here permanently to prevent repeated failures.

## Current State (as of 2026-06-11)

**E's structured partition:** `z_g` (50-d pose-invariant geometry) — sole survivor.
**Identity conditioning:** flesh-masked DINOv3 patch tokens (Phase 4, AUC 0.797, cross-shoot verified).
**DiT stack:** 2-stream — DINO patches (identity) + z_g (interpretable geometry control).

| Partition | Status | Key number | Detail |
|-----------|--------|------------|--------|
| z_g (geometry) | ✅ Survived | AUC 0.67–0.69 | Phase 1-R, shipped frozen encoder |
| z_d (depth) | ❌ Dead | ΔAUC −0.023 to −0.034 | Phase 2, confirmed at 24× resolution |
| z_a (normals) | ❌ Dead | ΔAUC −0.039 | Phase 2b, initial PASS overturned |
| DINO bridge | ❌ Dead | R² 0.385, transfer ≤ random | Phase 3, both directions dead |
| DINO masked patches | ✅ Survived | AUC 0.797 | Phase 4, settled identity carrier |

**Next target:** Phase 5 — DiT fusion stack.
**Key methodological lessons:** verification AUC > trace-J; every transfer gate needs a random-projection null; measurement-resolution baseline trap (confidence ≠ precision); seg-collapse detection (empty vectors poison gates).

---

## [Phase 1] Geometry PCA Encoder (`exp/geometry-pca`)

**Date:** 2026-06-09
**Goal:** Build a frozen, orthogonal Geometric PCA encoder (z_g) from 2D predictive keypoints (DWPose/COCO-WholeBody), validating that Generalized Procrustes Analysis (GPA) isolates morphology from camera perspective.

### Empirical Evidence
* **Data:** 70,000 samples from `stratum-ffhq`, sliced to 68 iBUG facial keypoints, confidence filtered (>= 0.5).
* **Retained Variance:** 99.93% variance retained at $k=50$ components.
* **Reconstruction RMSE:**
  * 1 PC: 0.0082
  * 10 PCs: 0.0018 (The "elbow" of the scree plot)
  * 50 PCs: 0.00037 (Resolves down to pixel-level asymmetry)

### Deconstruction of Latent Traversals
Visual analysis of the $\pm3\sigma$ traversals confirmed complete decoupling of perspective and morphology:
* **C1 (Horizontal Yaw):** Smoothly transitions the head from looking right to left.
* **C2 (Vertical Pitch):** Captures head nodding up and down.
* **C3 (Global Aspect Ratio):** Pure morphology. Isolates facial width (ectomorphic vs endomorphic).
* **C4 (Mouth Opening):** Localized action. Translates the jaw and lower lip while eyes/nose remain strictly static, confirming the orthogonality guarantee.
* **C5 (Upper-Face Modulation):** Eyebrow arching and orbital spacing.

### Verdict
**PASS (REVOKED — see Phase 1-R below).** The mathematical firewall operated
exactly as asked. But on review we determined we asked the *wrong question*:
the encoder is pose-*entangled*, not pose-*invariant*, which disqualifies it as
an identity descriptor. Verdict revoked 2026-06-09.

---

## [Phase 1-R] Geometry Encoder REOPENED — pose-invariance mandate

**Date:** 2026-06-09
**Trigger:** Post-PASS review of the C1/C2 traversals.

### Why the PASS was revoked (the question was wrong, not the math)
The Phase 1 traversals proved C₁ = yaw and C₂ = pitch — clean, but **fatal**:

1. **Semantic category error.** `E`'s North Star is to describe the *invariant
   person*. Yaw/pitch are transient camera state, not identity. An identity
   vector that encodes orientation is mathematically disqualified.
2. **Why 2D GPA could never fix it.** 2D GPA neutralizes only 2D transforms
   (translation, scale, in-plane roll). It is **blind to out-of-plane 3D rotation
   (yaw/pitch)**. PCA therefore correctly shoved that dominant variance into the
   top components. The pipeline did its job; the objective was mis-specified.
3. **Double-conditioning conflict.** The DiT already ingests raw `pose.npy` as
   the authoritative orientation signal. A pose-encoding `z_g` conditions the
   same fact down a second, conflicting path → optimization conflict. `z_g` must
   be the **pose-orthogonal complement** of `pose.npy`.

### Decision
- **Mandate:** pose-invariance by construction (01_VISION_AND_ARCHITECTURE.md §3.2 updated).
- **Rejected — Frontal Filter (data solution):** dropping non-frontal samples
  also discards the profile-only biological signal (nose projection, jaw/brow
  depth) that the North Star explicitly wants. Narrows the input distribution
  instead of factoring out the nuisance variable. Lossy where it hurts most.
- **Chosen — 3D-aware alignment (algebraic solution):** estimate head rotation,
  rotate to a canonical frontal 3D frame, reproject, then PCA. Preserves
  extreme-angle samples; the reprojected X-spread encodes depth signal.
- **First step — EPnP SPIKE (cheap, deterministic):** PnP rotation of the 68
  points against a canonical 3D mean-face template → reproject → re-run the
  existing PCA + traversal gate, plus a new synthetic-pose-invariance probe.
  Changes exactly one variable. If C₁ cleans up to morphology, thesis proven;
  escalate to a full 3DMM (morphometrics repo) only if the spike falls short.

### Status
`[CONCLUDED]` — EPnP spike graduated to production encoder; 3D-frontalized z_g shipped.

### Spike Result (2026-06-09) — THESIS PROVEN
Pose-normalization (orthographic-PnP rotation estimate from the 68 keypoints
against a canonical 3D template → lift-to-3D using a depth prior → rotate to
frontal → reproject → PCA) was run against the *raw* Phase 1 encoder, changing
only the alignment variable.

**Quantitative (pose-invariance probe):** one identity, yawed ±30°, encoded by
each model. Mean per-component std of `z_g` across the synthetic-yaw set:
* Raw Phase-1 encoder: **1.45** (yaw leaks ~1σ into the sliders)
* Pose-normalized encoder: **0.29**
* **Improvement: 4.9× more pose-invariant.** Variance retention unchanged (99.94%).

**Visual (traversal gate):** the new C1 and C2 stay **frontal and bilaterally
symmetric** across ±3σ — C1 now reads as face width / aspect-ratio morphology,
C2 as upper-face/brow structure. The lateral nose-vs-jaw shear (yaw) and vertical
whole-face compression (pitch) are gone. See
`docs/assets/exp/geometry-pca/posenorm_traversal_C{1,2}.png`.

**Verdict:** EPnP-style frontalization is sufficient. **No full 3DMM needed.**
Recommend graduating the `pose_normalize` step into the canonical encoder
pipeline (replacing plain 2D GPA as the first alignment stage). The lightweight
orthographic solver is deterministic, CPU-only, and scales to 70k in minutes.

**Caveats / follow-ups before final sign-off:**
* Spike ran on a 2k subset; re-run on the full 70k to confirm at scale.
* Depth prior is a hand-built neutral radial profile; a data-driven depth
  template (or a real iBUG 3D reference) would sharpen frontalization further.
* Probe used synthetic yaw built from the same depth prior — somewhat
  self-consistent; an independent yaw source (real multi-view) would be stronger
  proof. Acceptable for a spike; note for the productionized gate.

---

## [Phase 1-R FINAL] Production close — z_scale=1.0, real-image gate, CONCLUDED

**Date:** 2026-06-10

### The synthetic probe was abandoned (it was circular)
The spike's apparent "4.9× pose-invariance win" was partly self-fulfilling: the
synthetic-yaw probe lifted 2D points with a depth model and then frontalized them
with that *same* depth model — so a FLAT template (z_scale=0) scored best by
trivially bypassing the Z-axis. It was a test of mathematical reversibility, not
of biological pose-invariance. **We abandoned the synthetic probe entirely** and
built a real-image gate.

### The contamination near-miss (PERMANENT WARNING)
We sourced real multi-pose identities from the hegre dataset (model name in the
folder slug). The FIRST gate run (5 identities) returned J≈0.085 with FLAT
(z_scale=0) marginally winning → it pointed at "kill the 3D pipeline." **This was
an artifact.** Visual collage inspection (the check that saved us) revealed 4 of 5
"identities" were contaminated:
* `darina` merged TWO women — a brunette (`darina-*`) and a blonde (`darina-l-*`).
  Name-collision: the `-l`/`-s` suffix denotes a *different model*.
* `ariel`, `valerie`, `emily` each contained a MALE partner's face, pulled from
  couple shoots (`-and-`) by DWPose `single_person=True` grabbing the largest bbox.

The contaminated within-identity scatter was a LABELING artifact, not a property
of geometry. **The artifact-driven data nearly caused us to amputate a
mathematically sound 3D frontalization step.** Lesson logged permanently: never
trust an identity gate without visually verifying the identities; never trust a
synthetic probe that can reverse its own math.

### Clean re-run (10 verified identities, 136 real images)
Fixes: suffix-aware identity keys (`darina-l` ≠ `darina`), couple-set exclusion
(`-and-`/`couple`), and per-identity collage verification (dropped `muriel` =
male+blur, `natalia-a` = ambiguous). Fisher S_B/S_W sweep over z_scale:

| z_scale | J global | S_B  | S_W   | J_C1  |
|---------|----------|------|-------|-------|
| 0.00 (flat/2D GPA) | 0.0800 | 25.1 | 313.9 | 0.0655 |
| 0.50    | 0.0701 | 19.6 | 279.8 | 0.0863 |
| **1.00 (SHIPPED)** | 0.0868 | 21.3 | 245.8 | 0.0795 |
| 2.00    | 0.0877 | 20.4 | 232.8 | 0.0782 |

### Findings (stated honestly)
1. **3D frontalization beats flat 2D GPA on real-image identity separability.**
   Global J rises 0.080→0.088 with depth, driven by within-identity scatter
   S_W falling 314→246 while S_B holds — the OPPOSITE of degenerate collapse.
   3D pose-normalization systematically strips pose variance across the manifold.
2. **The clean-C1 narrative DIED at scale.** At n=5, J_C1 rose monotonically with
   depth; at n=10 it did NOT (peaks at z=0.5, noisy). We do NOT claim a clean
   C1-rescue. Once macro yaw variance is stripped, PCA promotes whatever residual
   chaos remains (expression asymmetry, focal distortion, DWPose jitter) into the
   top component — that is the mathematical reality of wild 2D tracking, not a
   depth-model failure. **The case for 3D rests on AGGREGATE S_W reduction.**
3. **Absolute separability is modest (J≈0.08; S_W ≈ 12× S_B).** This is a FEATURE,
   not a bug: it is the empirical proof that 2D facial geometry alone is a noisy
   standalone identity carrier under real-world pose/expression. It directly
   justifies the multi-partition E = [z_g | z_d | z_a] structure — if geometry
   were a perfect identity carrier, the depth/normals/DINOv3 partitions would be
   bloat. Editorial-data caveat: hegre shoots vary in expression/lighting/age, so
   S_W is inflated by non-pose nuisance (makes the test strictly harder).

### z_scale = 1.0 decision (anatomical mandate)
Shipped z_scale=1.0, NOT the marginally-higher-J z=2.0 nor the C1-peak z=0.5.
Rationale: 1.0 uses the 300W canonical template's depth at face value (physical
ground truth); scaling down tells the solver a face is a pancake, scaling to 2.0
extrapolates depth beyond anatomy for a negligible J gain. z=1.0 captures ~85% of
the total S_W reduction without extrapolation — the production sweet spot.

### Production artifact
Frozen encoder fit on the FULL **69,851** FFHQ faces (k=50, 99.987% variance,
107s total). Pipeline: pose → 68-pt slice → **mean-confidence prefilter (drop
faces with mean DWPose confidence < 0.5)** → 3D frontalize (canonical 300W
template, z_scale=1.0) → light 2D GPA → PCA → whiten. Canonical template
persisted in the encoder for reproducible inference. Artifact:
`output/encoder_production.npz` (verified contents: `components` (50,136),
`canonical_template` (68,3), `pca_mean`, `whiten_mu`/`whiten_sigma`, `gpa_mean`).

### Verdict
**Phase 1-R CONCLUDED — PASS (earned).** Pose-invariant geometry encoder shipped.
Honest scope: 3D frontalization gives a real aggregate identity-separability gain
over 2D GPA; the clean-C1 story did not survive scale; geometry alone is a weak
identity carrier (motivating the rest of E).

> **Evidence-file caveat (2026-06-10):** the machine-readable sweep artifact
> `docs/assets/exp/geometry-pca/gate_sweep_results.json` is **truncated/corrupt**
> on disk (dies at byte 187, inside the first `z_scale=0.0` result row; only the
> top-level `best_z_scale`/`best_J`/`null_J_flat_2dgpa`/`3d_beats_flat` scalars
> are readable). The **authoritative Phase-1-R sweep values are the table above**
> (§Clean re-run). We are NOT regenerating it: `07_gate_sweep.py` hard-codes the
> legacy 10-identity gate (`FIT_LIMIT=5000`, `data/hegre_gate_keypoints.npz`,
> `DROP={muriel,natalia-a}`), so a re-run reproduces the *historical* 10-id
> result, not today's expanded set. The next clean machine-readable gate artifact
> will be produced by the Phase 2 gate (below), which runs on the full reviewed
> identity set. Note also the script's internal "beats flat" threshold is ×1.05;
> the Phase 2 incremental-information bar is a deliberately stricter ×1.15.

---

## [Phase 2] Volumetric Encoder z_d (depth) — `[CONCLUDED — FAIL]`

**Date opened:** 2026-06-10
**Goal:** Build the depth partition `z_d` of `E = [z_g | z_d | z_a]` from
`depth.npy` (Sapiens), and prove that depth carries **complementary identity
signal beyond 2D geometry alone**.

### Pre-registered gate (stated BEFORE results — honest-science discipline)
> **PASS criterion:** `J([z_g | z_d]) > J(z_g) × 1.15`
> on the hegre identity-separability test (Fisher S_B/S_W).

This is an **incremental-information** test, not an absolute-separability test:
concatenating the depth partition onto the geometry partition must lift the
Fisher discriminant ratio by **at least 15%** over geometry alone. If depth is
redundant with 2D geometry, J will not move and the partition is bloat; the
×1.15 bar (stricter than the Phase-1 ×1.05 "3D-beats-flat" threshold) forces
depth to earn its place in `E`.

### Identity test set (canonical, growing)
The gate runs on the reviewed **hegre** corpus in
`experiments/geometry_pca/data/review.db` — **not** the legacy 10-identity set.
Current snapshot (2026-06-10): 120 personas / 2400 images reviewed →
**1,524 `approved` images across 89 contamination-free identities**. Exclusion
rule (from the review system): any persona with ANY `tainted:contamination`
image is dropped entirely from the gate. The corpus is **growing in the
background** — more personas (breadth) and more images per persona (depth) — so
the gate must be re-runnable as `review.db` expands; the 89/1,524 figure is a
snapshot, not a frozen N.

### What is already BUILT (verified on disk)
- **Depth preprocessing** (`scripts/18_fit_zd_encoders.py`, commit `3a3793a`):
  seg-mask → face-crop → canonical resample, with **3 normalization modes**
  (A / A_prime / C) to be gated against each other.
- **Single-pass NAS depth cache** (`scripts/19_build_depth_cache_singlepass.py`,
  commit `91b527f`): collapses the old 6-NAS-pass design (3 modes × 2 passes)
  into 1, writing `data/depth_cache/ffhq_depth_{A,A_prime,C}.npy` + `ids.json`.
  Storage-rule compliant — `data/` is a symlink to the NAS project folder.
- **z_d encoder fit** scaffolding (`18_fit_zd_encoders.py`) writing to `output/`.

### What is OPEN (the actual work remaining)
1. Decide/gate the depth normalization mode (A vs A_prime vs C).
2. Fit the frozen `z_d` PCA encoder on FFHQ depth (k≈50, whitened).
3. Build a `z_d` gate extractor over the `review.db` approved set (analog of
   `06_extract_hegre_gate.py`, but reading depth + driven by the DB, not the
   legacy `.npz`).
4. **Run the gate**: compute `J(z_g)` baseline and `J([z_g | z_d])`, check ×1.15,
   write a fresh (non-corrupt) machine-readable results artifact.

### Normal-map / z_a note
**The z_a pivot was attempted (2026-06-10) but later overturned (2026-06-11).**
Normals structurally avoided the affine-scale ambiguity that killed z_d, but the
initial PASS was an artifact of the low-resolution z_g baseline. See
**[Phase 2b]** entry below for the full story.

### Verdict
**[z_d CONCLUDED — FAIL (above). z_a CONCLUDED — FAIL (below, initial PASS overturned).]**

---

### [Recovered from Orphaned Section] The Evidence for Phase 2's Failure
The depth partition `z_d`, as currently encoded (64×64 masked resample, k=50,
FFHQ-fit basis), adds **no usable complementary identity signal** on top of `z_g`.
This is a high-value negative result, established after isolating and correcting a
metric bug — the two are independent and both are recorded below.

#### Gate run (102 clean identities, 1,665 images from review.db)
| | trace-J | vs z_g |
|---|---------|--------|
| z_g baseline | 0.092 | — |
| best z_d mode (A_prime, re-std) | 0.094 | **×1.02** |
| mode C (re-std) | 0.090 | ×0.98 |

Raw `J([z_g\|z_d])` ×1.02 — far short of the pre-registered ×1.15 bar. **FAIL.**

#### ⚠️ Metric bug found during review: trace-J cannot measure complementarity
The gate used the **trace** Fisher ratio `J = tr(S_B)/tr(S_W)`. For a *concatenated*
vector the scatter traces decompose additively, so:

```
J_cat = ( tr(S_B,g) + tr(S_B,d) ) / ( tr(S_W,g) + tr(S_W,d) )
```

This is **exactly a weighted average** of `J_zg` and `J_zd` (weights = S_W shares).
If `J_zd` < `J_zg` (which it is: 0.03 vs 0.09), appending z_d will **always pull
down the trace average**, even if the components are perfectly orthogonal and
contain 100% independent identity signal. The metric mathematically guarantees
failure for any partition weaker than the strongest one.

*Correction:* Switched the canonical gate instrument from trace-J to
**verification AUC** (same/different identity discrimination via cosine distance
of the concatenated vector).

---

## [Phase 2b] Surface Normals Partition z_a — `[CONCLUDED — FAIL]`

**Date opened:** 2026-06-10
**Goal:** Build the surface partition `z_a` from Sapiens normal maps and prove
normals carry **complementary identity signal beyond 2D geometry**, where depth
(z_d) failed. Normals describe surface *angle*, not absolute distance, so they
natively resist the affine-scale / camera-distance ambiguity that killed z_d —
this is the pivot thesis.

### Why normals should beat depth (the structural advantage)
- **No scale ambiguity.** Sapiens normals are unit vectors on the foreground
  (probed: ‖n‖=1.0000 on both FFHQ and hegre), so there is no camera-distance or
  focal-length variable to corrupt the signal. z_d's A/A_prime/C normalization
  sweep has **no analog here** — every variant is unit-norm by construction.
- **Pose = the real nuisance.** Head rotation coherently rotates the entire
  normal field. But we already own the antidote: Phase 1-R's
  `estimate_rotation()` gives per-sample head rotation R from the 68 keypoints.
  De-rotating normals by Rᵀ puts them in a canonical head frame — the
  normal-space equivalent of 3D frontalization.
- **Redundant channels.** Visible surfaces face the camera (nz>0), so
  nz = √(1−nx²−ny²) is redundant — an (nx,ny)-only variant halves the
  dimensionality for free.

### Representation sweep (replaces z_d's normalization sweep)
| Variant | Channels | Rationale |
|---|---|---|
| `raw`    | (nx,ny,nz) 64×64×3 = 12,288-d | naive baseline |
| `xy`     | (nx,ny)    64×64×2 = 8,192-d  | nz redundant for camera-facing surfaces |
| `rot`    | Rᵀ·n, 3ch  12,288-d             | head-pose de-rotation → canonical head frame |
| `rot_xy` | Rᵀ·n, xy   8,192-d              | both corrections |

Tangent-space log-map deferred — only if all 4 Cartesian variants fail marginally.

### Identity test set (same as z_d)
The gate runs on the reviewed **hegre** corpus in `data/review.db` (READ-ONLY
— sole writer is Tim's validation window). Current snapshot: 102 clean identities
/ 1,665 approved images. **Same set as z_d** for comparability.

### Pre-registered gate (stated BEFORE results — honest-science discipline)
> **PASS criterion:** mean over seeds {0,1,2}:
> `AUC([z_g | z_a]) > AUC(z_g) + 0.01`
> on the hegre verification test (same/different-identity discrimination,
> cosine distance, z-scored, balanced pairs, n=40k/seed).

- **Metric:** verification AUC (canonical instrument; trace-J banned for
  concatenated partitions — see [Metric fix] above).
- **ε = 0.01** ≈ 4× the measured seed noise (±0.0025 from z_g baseline).
- **Secondary report (not pass/fail):** z_a-ALONE AUC — if normals alone ≫ 0.54,
  they are a stronger standalone identity carrier than geometry.
- **Variant selection:** by highest mean AUC delta among variants that pass
  — **not** by retained variance (the z_d lesson).
- **Nuisance audit (before accepting a PASS):** correlate top-5 z_a components
  vs estimated yaw/pitch per image — the z_d C1-audit, run BEFORE trusting
  the gate, not after.

### Architecture decisions
- **k = 50** — partition-size consistency with E = [z_g|z_d|z_a] ∈ ℝ^150.
- **One NAS pass, one cache** (raw grid + per-sample R → 4 variants in RAM).
- **Pooled vectors not renormalized** — the sub-unit magnitude after pooling
  IS local curvature disagreement (signal).

### Verdict
**[CONCLUDED — FAIL (initial PASS overturned 2026-06-11)].**
> ⚠️ The verdict below was overturned by the face-crop re-run. See
> **[2026-06-11 UPDATE] Phase 2b (Normals) Face-Crop OVERTURN** at the end of
> the Phase 2 section.

The initial PASS (2026-06-10) appeared to show surface normals carrying
complementary identity signal beyond geometry, where depth failed. This was
later found to be an artifact of the artificially low editorial-keypoint z_g
baseline (0.540). When re-tested at proper face-crop resolution with the
corrected z_g baseline (0.688), normals *subtract* identity signal
(ΔAUC −0.039). **See overturn at line 571.**

#### Gate run (102 identities, 1665 images)
| Variant | z_a alone AUC | [z_g\|z_a] Δ | vs ε=0.01 | Nuisance |
|---|---|---|---|---|
| `rot` | 0.570 | +0.0283 | ×2.8 | SUSPECT |
| `raw` | 0.567 | +0.0267 | ×2.7 | SUSPECT |
| `rot_xy` | 0.567 | +0.0265 | ×2.7 | SUSPECT |
| `xy` | 0.562 | +0.0237 | ×2.4 | **CLEAN** |

**z_g baseline: 0.540** (chance=0.5).

**Key findings (⚠️ SUPERSEDED):**
*The findings below reflect the 2026-06-10 data. They are preserved for provenance, but their conclusions were invalidated by the 2026-06-11 face-crop re-run.*

1. **[INVALIDATED] z_a ALONE beats z_g ALONE.** Every normal variant's standalone AUC
   (0.562–0.570) comfortably exceeds geometry's (0.540). Normals are a stronger
   identity carrier than frontalized 2D keypoints on editorial photos.
   *(Correction: The 0.540 baseline was a resolution artifact. Real z_g AUC is 0.688.)*
2. **[INVALIDATED] Complementary lift:** appending z_a to z_g lifts AUC by +0.023 to +0.028,
   clearing the pre-registered ε=0.01 bar by >2.4× in every mode. (This was
   re-verified across 10 seeds: worst single-seed delta was +0.021.)
   *(Correction: When tested against the true 0.688 baseline, z_a subtracts -0.039. It is not complementary.)*
3. **The rot paradox (visibility bias).** The `rot` variant nominally won, but
   the audit flagged it SUSPECT (high pose correlation), while `xy` was CLEAN.
   A mechanistic follow-up proved why: raw normals' mean direction is always
   camera-facing (pose-blind, visibility bias); applying Rᵀ de-rotation rotates
   that camera-facing mean, *injecting* head pose into the global mean direction
   of the grid. PCA promotes this variance. Thus, de-rotation removes pose from
   the texture but injects it globally.
4. **[INVALIDATED] Architectural choice:** **`xy` (8192-d)** is the canonically selected
   variant. It is CLEAN of pose nuisance, requires zero de-rotation (avoiding
   the visibility-bias paradox), is the most compact representation, and passes
   the gate cleanly in 10/10 seeds.

### Artifacts
- Encoders: `output/encoder_za_{raw,xy,rot,rot_xy}.npz`
- Gate results: `data/za_gate_results.json`
- Systematic review: `data/za_systematic_review.json`
- Scripts: `24` (cache), `25` (fit), `26` (extractor), `27` (AUC gate),
  `28` (systematic review).

---

---

## Phase 5a: Semantic Geometry Mapper (text-to-zg)

**Opened:** 2026-06-24

**Goal:** Train an offline MLP to predict canonical geometry ($z_g \in \mathbb{R}^{50}$) from semantic text embedding (T5) and identity (AuraFace).

**Premise:** Text descriptions of geometry ("sharp jawline") are relative to base identity. An AuraFace identity vector resolves this ambiguity. However, because single-image AuraFace embeddings leak head pose (Tier 0.2), we must use the **persona-averaged AuraFace vector** as the identity anchor. This marginalizes out transient pose and expression noise, forcing the MLP to rely on the textual semantics to interpret the geometry.

**Hypotheses:**
* **$H_0$:** A model predicting $z_g$ from `[T5 || \overline{AuraFace}]` does not significantly decrease validation MSE compared to `T5` alone.
* **$H_1$:** Identity conditioning resolves semantic ambiguity, resulting in a statistically significant reduction in validation MSE.

**Methodology:**
1. Enrich Hegre dataset with `caption` and `t5` passes via LLaVA/Ollama.
2. Build dataset mapping `T5_image` to `[\overline{AuraFace}_{persona} || z_g]`.
3. Train baseline (T5-only) vs Conditioned MLP and evaluate on a held-out test split.

**Status:** `[ACTIVE]` — Enrichment pipeline patched, dataset builder written. Waiting on VLM captioning run.

## [Phase 3] DINOv3 Bridge (Premise Validation) — `[CONCLUDED]`

**Date opened:** 2026-06-10
**Goal:** Linear-regress DINOv3 semantic embeddings (`dinov3_cls`, 1024-d) to the
whitened physical sliders (`z_g`, `z_a`).

### Pre-registered gates (stated BEFORE results — 2026-06-10)

**Phase 3 (The Premise Test — FFHQ)**
Fit via 5-fold CV Ridge Regression.
* **PASS:** Variance-weighted held-out R² **≥ 0.5**, AND per-component R² **≥ 0.6
  for C1–C10** (coarse structure).
* **Falsifiable prediction:** `z_a` (micro-surface) will have a strictly lower R²
  spectrum than `z_g` (coarse geometry), as the 16x16-patch DINO token discards
  fine curvature.
* **Diagnostic band:** If 0.25 ≤ R² < 0.5, run a 2-layer MLP probe to test if the
  mapping is merely nonlinear.

**Phase 3b (The Transfer Test — hegre)**
If Phase 3 passes, apply the FFHQ-fit bridge `W` to hegre editorial photos to get
predicted sliders `Ŷ_a`. Run the canonical verification-AUC identity test.
* **PASS:** `AUC(Ŷ_a) > 0.5 + 4σ_seed (≈0.51)`.
* Proof that the bridge preserves *identity*, not just variance, under domain shift.

### Stratified Verdict (CORRECTED after systematic review — script 32)

**Phase 3 (The Premise Gate — FFHQ): `[FAIL]`**
*   `z_g` (Geometry): Variance-weighted R² = **0.690**. FAIL on the C1–C10 ≥ 0.6
    criterion: C6 (R²=0.023) & C11 (R²=0.017) are near zero.
    **Caveat (C4 probe):** real-z_g C6 has Fisher J = 0.098 on hegre — at the
    median of all components — so C6 is plausibly *detector noise that is
    intrinsically unpredictable*, not identity-critical structure DINO "missed."
    The C1-C10 criterion presumed all top-10 components are semantically
    meaningful; that presumption was partly wrong.
*   `z_a_xy` (Surface): Variance-weighted R² = **0.385** (MLP probe: 0.48). FAIL.
    Genuine reconstruction failure — the DINO cls token does not retain fine
    surface curvature.
*   Verified: independent 80/20 refit reproduces both R² (0.692 / 0.385); proper
    pre-fit label-shuffle null ≈ −0.015 (no leakage). NOTE: the original
    script-30 "permutation null" shuffled Y *after* prediction (analytically
    = −R², vacuous); fixed in the same review.

**Phase 3b (The Transfer Gate — hegre): gate technically passed — but the PASS
is UNINFORMATIVE (missing control, caught in review).**
*   *Note: Re-run on cropped `hegre_faces_stratum` to remove scene-level noise.*
*   Measured: Ŷ_a AUC 0.674, Ŷ_g AUC 0.704 (FFHQ-fit bridge).
*   **The control that kills the story:** AUC(raw dinov3_cls face crop, 1024-d) = **0.766**;
    AUC(random Gaussian 50-d projections of DINO) = **0.712 ± 0.007** (5 seeds).
    The bridge (0.704) is *worse than a random projection* of its own input.
    Any 50-d DINO shadow clears the 0.51 bar → the pre-registered gate was
    structurally too weak (it lacked the random-projection null).
*   **Domain-shift vs Projection loss:** A 5-fold CV hegre-fit bridge control
    yielded Ŷ_g AUC 0.673. The failure is *not* FFHQ→hegre domain shift; Ridge
    regression mathematically destroys identity when forced to map to physical
    geometry, even on the target domain.

### The real findings of Phase 3
1.  **Faithful slider reconstruction from DINO is dead** (both directions of the
    "fast path"). E cannot be derived from DINO embeddings without fatal
    identity loss.
2.  **Raw `dinov3_cls` face crops are the strongest identity carrier measured:
    AUC 0.766** — above face-crop z_g (0.67–0.69; see Phase 2 [2026-06-11
    CORRECTION] — the older "z_g 0.540" was an editorial-keypoint-resolution
    artifact) and far above z_d (0.56).
    *Caveat (C5 Shoot-Leakage):* DINO's "identity" includes same-shoot lighting/
    background recognition (same-shoot sim 0.63 vs cross-shoot 0.19).
    However, for the DiT, DINO tokens are the natural primary *identity*
    conditioning.
3.  **E's unique, irreplaceable value is interpretable decoupled control.**
    Since DINO cannot faithfully reconstruct E's components, E remains structurally
    non-redundant.
4.  **Lesson (gate design):** every transfer/identity gate must include a
    random-projection null of its input representation, exactly as every
    partition gate includes a permutation null. A gate without the right null
    can "pass" on structure the test never isolates.

### Artifacts
- Bridge weights: `output/bridge_dinov3.npz` (kept for reference; NOT a product)
- Phase 3 R² results: `data/phase3_bridge_results.json`
- Phase 3b AUC results: `data/phase3b_transfer_results.json`
- Systematic review: `data/phase3_systematic_review.json` (script
  `32_phase3_systematic_review.py`: R² cross-check, proper null, two-tree
  alignment 1721/1721, dup/finite checks, raw-DINO + random-projection
  controls, C6-noise probe, shoot-leakage probe)

|--------|----------------------|-------------|---------|
| trace-J | ❌ (the bug) | ×1.02 | flat — the weighted-avg trap |
| multivariate-J `tr(S_W⁻¹S_B)` | ✅ but inflates w/ K=100 | ×1.7 | **suspect** (dimensionality) |
| kNN identity accuracy | ✅ operational | −0.2% (4.3%→4.1%) | **no help** |
| **verification AUC** | ✅ decisive, bias-immune | **−0.004** | **no help** |

The two *operational* tests (can we actually identify the person?) both say depth
adds nothing — in every mode. The tempting ×1.7 multivariate-J rise was a
dimensionality mirage, refuted by the operational metrics disagreeing with it.

#### Secondary finding (quantifies Phase 1-R)
**z_g's own verification AUC = 0.541** (chance = 0.5; stable 0.538–0.543 across
seeds). Frontalized 50-d facial geometry is a *very weak* identity discriminator on
hegre editorial photos — an operational quantification of Phase 1-R's "geometry
alone is a weak carrier (J≈0.08)". Depth at 64×64/k=50 not only fails to help, it
slightly *dilutes* this already-weak signal (AUC −0.004).

#### What is NOT ruled out
"Depth as currently encoded is a dead end" — NOT "depth is useless". Untested rescue
levers, now measurable with the sensitive verification-AUC instrument: higher
resolution (>64px), more components (k>50), a hegre-fit basis (not FFHQ). But raw
*monocular relative depth* is fundamentally entangled with camera distance/focal
length (affine-scale ambiguity), so these fight uphill.

#### [2026-06-11 UPDATE] Face-crop re-run: FAIL confirmed at 24× facial resolution
The resolution + domain-shift rescue levers above are now TESTED, via the new
`hegre_faces_stratum` dataset (Sapiens enrichment run natively on face crops):
facial depth resolution ~695² px vs ~140² inside editorial frames (**24× more
facial depth detail**), and the FFHQ-fit whitening now lands near-in-distribution
on the gate set (whitened z_d per-comp std ≈1.0–1.3, vs 1.97 on editorial — the
domain shift is gone). Gate extractor `20` remapped to the face tree.

| | full face-crop set (1,448 img / 101 ids) | seg-clean subset (1,351 / 100) |
|---|---|---|
| z_g baseline AUC | 0.6813 | 0.6892 |
| best z_d-alone AUC | 0.5638 (A) | 0.5553 (A) |
| best [z_g\|z_d] delta | −0.023 (C) | −0.034 (C) |

**FAIL is robust — all modes, both sets: concatenating z_d *subtracts* identity
signal.** With resolution and distribution exhausted as excuses, monocular
relative depth at k=50 is conclusively a dead partition; only k>50 remains
untested and is not worth pursuing against uniformly *negative* deltas.

**Data defect found & controlled during this review (seg-collapse on face crops):**
Sapiens body-part segmentation collapses on ~10% of tight face crops (seg
foreground <30%; 7.5% fully empty at <2%) — visually confirmed perfect frontal
faces yielding ~99%-empty seg-masked depth maps. These "empty depth" vectors
created cross-identity near-duplicates in z_d (e.g. gislane≈vika cos 0.994).
The gate was re-run excluding all 97 affected rows: the FAIL *strengthened*
(−0.023 → −0.034), proving the defect was not masking a PASS. ⚠️ `normal.npy`
is masked by the same seg — **Phase 2b (z_a) on face crops MUST apply a
seg-foreground filter (≥30%)**. Control script: `36_zd_facecrop_seg_control.py`.

#### [2026-06-11 CORRECTION] z_g was understated: keypoint-resolution artifact
The "Secondary finding" above (z_g AUC = 0.541) is an artifact of editorial-frame
*keypoint measurement resolution*, not a property of facial geometry. Controlled
comparison — same 1,429 images, same frozen production encoder, only the DWPose
source differs:

| pose source | z_g AUC |
|---|---|
| editorial-frame DWPose (face ≈140px of frame) | 0.5405 |
| face-crop DWPose (face ≈695px of frame) | **0.6706** |

+0.13 AUC purely from keypoint precision. Note the trap: editorial keypoint
*confidence* was HIGHER (0.944 vs 0.865) — confidence ≠ precision. **Frontalized
facial geometry is a moderate identity carrier (AUC ≈0.67–0.69), not "very
weak".** The frozen encoder is unchanged; only its measured strength is
corrected. All gate baselines on face-crop data use the corrected z_g.

#### [2026-06-11 UPDATE] Phase 2b (Normals) Face-Crop OVERTURN
The initial PASS for surface normals (`z_a`) was a mirage caused by the artificially
low editorial-keypoint baseline. When re-tested on the proper `hegre_faces_stratum`
dataset (using the required seg-clean subset, fg≥30%), the gate failed decisively:

*   `z_g` Baseline AUC: 0.688
*   `z_a` (xy) alone AUC: 0.587
*   `[z_g | z_a]` AUC: 0.649 (**ΔAUC: −0.039, FAIL**)

**The Scientific Conclusion:**
Visual inspection confirmed the Sapiens depth/normals on these tight crops are
stunningly high-resolution, topologically accurate, and cleanly masked. Yet,
mathematically, they actively dilute the identity signal.
**Monocular volumetric models hallucinate generic, plausible human geometry.**
They do not encode the identity-specific biological micro-curvature required
for face recognition. The entire "fast path" (deriving decoupled structural sliders
from monocular networks) is a definitive dead end. 

**Eidolon's Conditioning Stack Simplifies:**
*   Identity: Raw DINOv3 Face Tokens
*   Interpretable Control: Geometry (`z_g`) ONLY.

---

## [Phase 4] Masked Patch Tokens (Semantic Face Isolation) — `[CONCLUDED — PASS]`

**Date opened:** 2026-06-11
**Goal:** The `dinov3_cls` token acts as a scene-level diplomat, forced to
summarize lighting, background, and clothing alongside the face. DINOv3's patch
tokens (`dinov3_patches`, 16x16 grid) are localized experts. By pooling ONLY the
patches that fall inside the Sapiens face mask (Masked Average Pooling), we force
the 1024-d identity embedding to care strictly about flesh, computationally isolating
the semantic identity from the shoot context.

### Pre-registered gates (stated BEFORE results)
The test is run on the clean face-crop set (1,460 imgs, 101 ids).

1. **The Representation Gate (AUC):**
   * **PASS:** Masked Patch Mean AUC > **0.766** (the raw `cls` face-crop baseline).
   * **Control:** Unmasked Patch Mean AUC. (To isolate the effect of *masking* vs
     the effect of *mean-pooling patches*.)
2. **The Shoot-Leakage Probe (C5 Gap):**
   * **PASS:** The Same-Shoot vs Cross-Shoot similarity gap must SHRINK compared
     to the `cls` baseline. (If AUC rises but the gap stays flat, we found more
     signal but not less lighting/background leakage.)

### Artifacts (Expected)
- Script: `37_dino_patch_face_pooling.py`
- Results: `data/phase4_patch_pooling.json`

### Verdict — `[CONCLUDED — PASS]` (2026-06-11)

Run on the seg-clean face-crop set (1,351 imgs / 100 ids; fg≥30%, conf≥0.5):

| Arm | AUC | Cross-shoot-only AUC |
|---|---|---|
| `cls` (baseline) | 0.7691 | 0.7679 |
| patch mean, unmasked (control) | 0.7828 | 0.7817 |
| **patch mean, flesh-masked** | **0.7975** | **0.7965** |
| patch mean, flesh+hair | 0.7993 | 0.7983 |

**Gate 1 (AUC > 0.766): PASS.** Both masked arms clear the bar. Effect decomposes
cleanly: mean-pooling patches beats CLS (+0.014) and flesh-scoping adds (+0.015).
**Statistically robust:** identity-level bootstrap (200 resamples) Δ(flesh−cls)
= +0.027, 95% CI [+0.014, +0.045], P(Δ≤0) < 0.005. Per-seed spread ±0.002.

**Gate 2 (C5 shoot-gap shrinks): directionally PASS, but underpowered AND moot.**
Gap 0.568 → 0.464 (flesh), but the dataset has only 41 same-id same-shoot pairs
(99/100 ids span multiple shoots) — too few to power the estimate. The decisive
replacement instrument: **cross-shoot-only AUC** (same-id pairs REQUIRED to come
from different shoots — leakage removed by construction) reproduces the full
ordering within 0.001. The +0.028 lift is pure cross-shoot identity signal, and
the standard verification AUC was never meaningfully shoot-inflated (same-shoot
pairs too rare to matter).

**Flesh vs flesh+hair:** +0.002 apart — within seed noise. **Flesh-only selected**
on principle: hair is the shoot-styled confound; the bump is not distinguishable
from noise and the leakage risk is structural. (flesh = Goliath classes
{2 face_neck, 23–26 lips/teeth/tongue}, 16×16 block-pooled mask, >0.5 threshold,
masked average pool → 1×1024.)

**Engineering verification (audited before accepting the result):**
- Stratum patch layout verified at source: `[CLS, reg×4, patches…]`, spatial from
  idx 5, row-major, RoPE-resized to bucket dims, no center crop. Grid-vs-count:
  0 mismatches across 1,577 leaves.
- Visual alignment proof: patch-PCA RGB grids render face/hair exactly where the
  seg mask places them (no transpose/mirror/offset).
- No noisy-mean trap: flesh patches per image min=100 / median=1,261.

**Product note:** the pooled 1×1024 vector is the *gate instrument*. For DiT
conditioning, prefer the unpooled masked patch tokens (~100–1,900 face tokens,
median ~1,261) via cross-attention; the pooled mean is the compact fallback.

**Identity conditioning for the DiT is settled: flesh-masked DINOv3 patch
representation (AUC 0.797, fully cross-shoot). Stack: DINO patches (identity) +
z_g (interpretable geometry control).**

---

### Artifacts
- Encoders: `output/encoder_zd_{A,A_prime,C}.npz`
- Gate (trace-J, deprecated): `data/zd_gate_results.json`
- Complementarity re-test: `data/zd_complementarity_diagnostic.json`
- **Verification AUC (decisive): `data/zd_verification_auc.json`**
- Scripts: `18` (fit), `19` (cache), `20` (extract), `21` (gate, trace-J),
  `22` (complementarity diag), `23` (verification AUC)

---

## [Metric fix] Gate instrument: trace-J → verification AUC

**Date:** 2026-06-10
**Trigger:** trace-J complementarity bug (above).

**Decision:** The canonical partition-gate metric is now **verification AUC**
(same/different-identity discrimination via cosine distance on z-scored vectors).
Rationale: scale-invariant, threshold-independent, and immune to the dimensionality
inflation that makes multivariate-J `tr(S_W⁻¹S_B)` untrustworthy at K≈100. trace-J
is retained ONLY as a legacy diagnostic; it must never again be the pass/fail
criterion for a *concatenated* partition. The re-stated gate for any partition `z_x`:

> A partition earns its place iff `AUC([z_g | … | z_x]) > AUC(baseline) + ε`
> on the hegre verification test (ε to be set from the AUC noise floor).

Status: `[CONCLUDED]` — instrument implemented in `scripts/23_zd_verification_auc.py`;
lifted into reusable `geometry_pca/verification.py` for subsequent gates.

---

## [Phase 5-prep] text-to-z_g data pipeline + z_g/AuraFace role split (`exp/text-to-zg`)

**Date:** 2026-06-27
**Goal:** Build the unified `(text, z_g, AuraFace)` training corpus for the
text-to-identity Prior, and settle the architectural roles of the two
conditioning streams empirically rather than by assertion.

### Data artifacts produced
* **z_g extraction** (`scripts/pipeline/extract_zg_and_averages.py`, idempotent,
  recomputes-on-rerun to stay synced with `review.db` curation):
  * FFHQ: 69,862 per-image z_g (50-d) on NAS at `ffhq/zg/`.
  * Hegre: 69,110–69,896 per-image z_g (approved-only); ~1,124–1,232 approved
    images had no z_g (failed/zero DWPose) and were skipped.
  * Per-persona centroids: 324 z_g averages, 323 AuraFace averages
    (`hegre-faces/v1/averages/`). **Orphan: persona `hera`** has a z_g centroid
    but no AuraFace centroid (all frames failed detection) — drop or repair.
* AuraFace already complete: FFHQ 69,960, Hegre 70,257 (all unit-norm, 0 bad).

### Empirical findings
1. **z_g ↔ AuraFace are orthogonal (R² ≈ 0).** Ridge regression z_g→AuraFace on
   8k FFHQ pairs: held-out **R² = −0.033** (worse than mean). The 50-d geometry
   basis explains ~none of the 512-d identity embedding linearly. → The two
   streams are genuinely complementary; the planned "project z_g out of AuraFace"
   step is **pointless** (nothing linear to remove) and was dropped.
2. **z_g carries almost no identity (Fisher-J collapse).** Recomputed the Tier 0.3
   morphology/transient split on the **full 69,110 / 323-identity** corpus
   (vs. legacy 1,448 / 101): global Fisher **J = 0.059**. Morphology axes (J>0.15)
   dropped from **27 → 6**; 22 legacy "morphology" axes fell into noise.
   **The legacy 27/23 split is retired** — it was small-N optimism.
   → **z_g is a geometry/pose control space, NOT an identity space.** Identity
   lives in AuraFace. (Caveat: corpus still curating — 62k `bad_geometry`,
   216k unreviewed — so within-person scatter may be inflated by DWPose noise;
   directional, not final.)
3. **AuraFace has no low-rank PCA structure.** Pooled PCA (140,217 vectors):
   PC1 = 2.08% var, flat spectrum, participation ratio ≈ 217/512. **PC1 is a
   domain artifact** (FFHQ vs Hegre, separation 2.05), not identity — must be
   projected out before identity analysis. → Unsupervised GANSpace-style sliders
   are NOT available in AuraFace.
4. **AuraFace identity sliders exist via supervised LDA.** LDA on 259 train
   personas (PC1 removed), tested on **64 held-out identities**:
   top-80 LDA dims recover **AUC 0.965** vs 0.969 full-512 (99.6% of power in
   ~80 dims); top-40 = 0.956; top-20 = 0.934. Generalizes to unseen identities.
   → Natural compressed target for the identity Prior (~64-d, not raw 512-d).
5. **LDA basis is global-in-direction but Hegre-scaled.** FFHQ projects onto the
   Hegre-fit basis without collapsing (random-pair cosine mean +0.004, std 0.167)
   but with **53% of Hegre's spread**. Usable, but carries a Hegre population prior.
6. **Top LDA axes are interpretable but demographic + nuisance-contaminated.**
   Visual contact sheet (`output/auraface_lda_axes.png`): LDA1 ≈ clean
   coloring/ethnicity axis (Asian/tanned/dark ↔ fair/blonde European); lower axes
   muddier, with **occlusion (mask), accessories (sunglasses), makeup, and lighting
   anchoring extremes** → nuisance is leaking into the discriminant basis.

### Verdict
**Settled architecture:** `z_g` = geometric/pose control (identity-blind);
**AuraFace (LDA-compressed)** = sole identity carrier. Streams kept separate
(decoupled cross-attention) with asymmetric CFG dropout on AuraFace as the
non-linear firewall backstop. **z_g/AuraFace orthogonality is now measured, not
assumed.** Two separate Priors (text→z_g, text→AuraFace-LDA).

**Open / next:** (a) ~~nuisance purification~~ — yaw pose leakage removed (R²=0.41→0,
committed as `auraface_preprocessing.py`); occlusion/lighting proxies remain
unaddressed (no labels); (b) ~~LDA refit~~ — basis re-measured on full 69k corpus,
generalizes to held-out identities (AUC 0.965), used as Prior 2 target;
(c) **Prior training** → see pre-registered Phase 5a below;
(d) fix `hera` orphan + investigate the ~1.1k missing-z_g approved images.

Status: `[ACTIVE]` — measurements in this session via `execute_code`; data on NAS.

---

## [SUPERSEDED — see Phase 5a-exp1 below] Phase 5a: Text-to-Identity Priors (`exp/text-to-zg`)

> NOTE: The `[CONCLUDED — PASS]` verdict in this section was REVOKED (defective
> metrics — see the REOPENED block). The authoritative conclusion is the
> `[CONCLUDED] Phase 5a-exp1` entry at the end of this file.

**Date pre-registered:** 2026-06-27
**Date concluded:** 2026-06-28
**Goal:** Train two Rectified Flow Matching Priors that map text (T5 embeddings) to
the conditioning vectors needed by the Phase 5 DiT — a 50-d geometric $z_g$ and a
compressed AuraFace-LDA identity vector — with held-out identity-generalizing quality.

### Pre-registration (preserved for provenance)
*(The original pre-registration text follows unchanged from the 2026-06-27 entry.)*
[...]

### Gate Results (2026-06-28) [REOPENED — gate metrics defective]

**WARNING: The metrics recorded below are defective and the PASS verdicts are revoked.**
1. **G1 Units Mismatch:** Per-dim MSE was divided by per-image variance (summed across 50 dims), inflating the ratio by ~50x. It also improperly mixed FFHQ predictions with Hegre variance (distribution shift).
2. **G2 Subspace Error:** Cosine was measured between the prediction and the *LDA reconstruction* of the target, not the raw AuraFace vector. Because all reconstructions share a mean offset, the cosine was mechanically inflated (self-cosine 0.40, cross-identity cosine 0.37).

The script `train_priors.py` has been updated to use the correct metrics (per-dim variance for G1, Verification AUC on raw AuraFace for G2).

**Legacy (Defective) Training Run:** Both Priors trained on FFHQ 63k (59,383 train / 10,479 held-out)
for 50 epochs, AdaLN-ResNet (12 blocks, 1024 hidden), Rectified Flow Matching,
AdamW with cosine schedule, batch 512, GPU (RTX 4090).

| Gate | Prior | Pre-train (step 0) | Final (step 50) | Threshold | Verdict |
|---|---|---|---|---|---|
| **G1** | z_g (text→geometry, 50-d) | MSE 2.24, ratio **0.022** | MSE 1.56, ratio **0.015** | < 1.0 | ❌ **REVOKED** |
| **G2** | AuraFace-LDA (text→identity, 64-d) | cosine **0.019** | cosine **0.564** | > 0.3 | ❌ **REVOKED** |

**G2 convergence:** cosine 0.02 (pre-train, FAIL) → 0.38 (epoch 1, PASS) →
0.55 (epoch 20, converged) → 0.56 (epoch 50). The model genuinely learns to
predict identity vectors from text: random predictions at initialization, rapid
improvement within one epoch, stable convergence after ~20 epochs.

**G1 convergence:** ratio oscillates 0.015–0.024, never approaching the 1.0
threshold. The pre-training baseline (0.022) already passes — the model's
prediction variance is small relative to the large within-person Hegre z_g
variance (σ²_w ≈ 104). MSE improves from 2.24 → 1.56 but the ratio barely
moves. **Interpretation:** T5 captions predict z_g modestly (pose, framing,
coarse expression), but most of z_g's 50 axes are fine-grained structural
detail not described by VLM captions. G1 passes technically but is not
diagnostic — the large σ²_w denominator masks the model's limited capacity
to predict the less text-describable components of z_g.

### Settled architecture

The Phase 5 conditioning stack is now defined on empirical evidence:

| Signal | Carrier | Dimension | Trained Prior |
|---|---|---|---|
| **Identity** | AuraFace → LDA-compressed | 64-d | text→LDA (G2 cosine 0.56) |
| **Geometry** | DWPose → 3D GPA → z_g | 50-d | text→z_g (G1 ratio 0.015) |

**Inference pipeline:** T5 text → Prior 1 (z_g) + Prior 2 (AuraFace-LDA) →
[z_g (50-d) | AuraFace-LDA (64-d)] → DiT `prx-tg` (separate cross-attention
streams, asymmetric CFG dropout). The 50 z_g axes serve as geometric sliders
at inference — traverse any axis to control pose/expression/proportion without
affecting identity.

### Corrected Gate Results (2026-06-28)

After fixing the two metric bugs, the corrected gates were evaluated on the FFHQ
held-out tail (2,000 / 1,999 samples) using the saved 50-epoch checkpoints:

| Gate | Metric (corrected) | Value | Threshold | Verdict |
|---|---|---|---|---|
| **G1** | per-dim MSE / per-dim FFHQ variance | **1.75** | < 1.0 | ❌ **FAIL** |
| **G2** | Verification AUC vs **raw** AuraFace | **0.575** | > 0.5 | ⚠️ **WEAK PASS** |

**G1 (text→z_g) FAILS — and is worse than the null.** Model per-dim MSE = 1.73;
FFHQ per-dim variance = 0.99. Predicting the global mean scores ratio 1.0 by
definition, so the FM model is ~75% *worse* than a constant mean predictor.
Confirmed negative: **text does not predict z_g.** This is informative — z_g is
pose/framing/camera detail that captions don't describe, so z_g should be supplied
by the user's slider/pose control at inference, NOT predicted from text. (The FM
model underperforms the mean because it samples from noise; with little learnable
signal, samples scatter around the mean rather than collapsing to it.)

**G2 (text→AuraFace) WEAK PASS.** AUC 0.575 (pos cos 0.220, neg cos 0.204, margin
0.016). Real but modest identity signal — the pre-train→trained jump confirms
genuine learning, but 0.575 is far from production-grade verification.

### Ceiling Test (2026-06-28) — where is G2's loss?

To attribute the G2 weakness, the **maximum achievable AUC** given the LDA
representation was measured by skipping the Prior entirely: take ground-truth LDA
coords → reconstruct → verify against raw AuraFace.

| Representation | AUC vs raw AuraFace | Reading |
|---|---|---|
| raw vs raw (sanity) | 0.9989 | harness correct |
| cleaned (PC1+yaw removed) vs raw | 0.9989 | **cleaning costs zero identity** |
| **GT LDA-64 reconstruction vs raw** | **0.9998** | **ceiling is ~perfect** |
| GT LDA-32 | 0.9765 | still strong |
| GT LDA-16 | 0.8658 | degrades |

**Decisive finding: the LDA-64 representation is NOT the bottleneck.** Its ceiling
is 0.9998 — virtually all verification-relevant identity survives the 64-d
projection + reconstruction. Therefore the entire G2 gap (0.575 achieved vs 0.9998
achievable) lives in the **Prior** — the text→LDA-64 mapping itself. This corrects
an earlier mis-reading: raw *cosine* on this manifold is compressed (self 0.40 /
cross 0.37), but *AUC* (rank-based) shows near-perfect separability.

**Re-ranked improvement levers (representation levers eliminated):**
- ❌ Increase LDA dims / predict richer 512-d target — pointless (64-d ceilings at 0.9998)
- ✅ **Conditioning quality** — mean-pooling the (512,1024) T5 sequence into one
  1024-vector smears identity-rich token detail. FFHQ captions are highly
  identity-centric (skin tone, hair, face shape, eye/nose detail per the Stratum
  prompt), so the token structure carries the signal that mean-pooling destroys.
  → strongest suspect.
- ✅ **Objective alignment** — add cosine/endpoint loss term (training optimizes FM
  velocity MSE but is gated on cosine direction).
- ✅ **FM stochasticity** — test a deterministic regressor baseline; if plain MLP
  regression beats the FM Prior on AUC, FM is the wrong tool for a near-deterministic
  text→identity mapping.

Status: `[REOPENED — G1 FAIL (informative), G2 weak pass; ceiling test localizes
loss to the Prior's text conditioning, not the representation]`.
Branch: `exp/text-to-zg`.

---

## [PRE-REGISTERED] Phase 5a-exp1: G2 Conditioning Fix (`exp/text-to-zg`)

**Date:** 2026-06-28
**Motivation:** Ceiling test proved the LDA-64 representation is near-lossless
(AUC 0.9998); the entire G2 gap (0.575 achieved) lives in the Prior's text
conditioning. Current Prior mean-pools the (512,1024) T5 sequence into one
1024-vector, destroying token-level identity detail that the richly
identity-centric FFHQ captions demonstrably carry.

**Hypothesis (H₁):** Conditioning on the full T5 sequence via cross-attention
substantially raises held-out verification AUC over mean-pooling.
**Null (H₀):** No improvement beyond noise → text→identity is information-ceilinged
(many-to-one), and weak AUC is correct calibrated behavior, not a model defect.

### Arms (same data, same held-out FFHQ tail)
| Arm | Conditioning | Model | Tests |
|---|---|---|---|
| A (baseline) | mean-pooled T5 (1024-d) | FM AdaLN-ResNet | reproduces AUC 0.575 |
| B (full-seq) | full T5 (512×1024) cross-attn | FM + cross-attn | conditioning hypothesis |
| C (regressor) | full T5 cross-attn | deterministic regressor | is FM the wrong tool? |

### Pre-registered gates (fixed before training)
- **G2′ (primary):** Verification AUC vs RAW AuraFace, held-out FFHQ tail.
  Credit H₁ only if Arm B beats Arm A by ≥ +0.05 AUC (identity-bootstrap CI excl. 0).
- **Attribute-consistency (secondary):** does predicted identity land closer to
  faces sharing the caption's attributes (skin tone/hair) than to random faces?
  Separates "can't match exact person" (expected/fine) from "can't match
  description" (real failure).

### Eliminated levers (per ceiling test — do NOT pursue)
- Increasing LDA dims / predicting richer 512-d target (64-d ceilings at 0.9998)
- Swapping the T5 encoder (stays until full-seq conditioning proven insufficient)

### Decision tree
- B ≫ A → conditioning was the bug; adopt cross-attn, proceed to DiT.
- B ≈ A, C ≫ A → FM wrong tool; switch to deterministic regression.
- B ≈ A ≈ C ≈ 0.6 → information ceiling; text→identity is many-to-one.
  Reframe: text = coarse identity region; fine identity from reference-image AuraFace.

Status: `[PRE-REGISTERED]` — gates fixed; no code written yet.
Branch: `exp/text-to-zg`.

---

## [CONCLUDED] Phase 5a-exp1: G2 Conditioning Fix — RESULTS (`exp/text-to-zg`)

**Date concluded:** 2026-06-29
**Verdict:** Experiment succeeded. Both hypotheses resolved with corrected metrics.

### Final results (corrected metrics, raw-AuraFace verification AUC)

3-arm run (30 epochs, full FFHQ 59,466 train / 10,494 held-out):
| Arm | Conditioning | Model | Verif AUC |
|---|---|---|---|
| A | mean-pool T5 | FM | 0.5825 |
| B | full-seq T5 cross-attn | FM | 0.6335 |
| C | full-seq T5 cross-attn | deterministic regressor | 0.6835 |

→ **B − A = +0.051** (full-seq conditioning beats mean-pool — hypothesis confirmed).
→ **C − B = +0.050** (deterministic regressor beats Flow Matching on exact-match AUC).

Arm C convergence run, stopped at **epoch 35** (the peak — see harness lesson below):
| Metric | Value | Reading |
|---|---|---|
| **Verification AUC** | **0.687** | peak; identical to the 80-epoch peak → ~0.69 is the true ceiling, NOT undertraining |
| pos cos / neg cos | 0.259 / 0.215 | margin 0.044 — weak raw separation, but AUC (rank-based) is the trustworthy number |
| **skin_auc** | **0.889** | strong — text steers identity to the correct skin-tone region |
| **hair_auc** | **0.712** | solid — hair direction real but weaker (AuraFace de-emphasizes hair) |

### What this establishes (keepers — high confidence)
1. **Text → coarse identity region works.** Verif AUC 0.687, well above chance, clean plateau.
2. **The region is attribute-correct.** skin 0.889 / hair 0.712 — prediction matches the *described* traits. This is the metric that matters for a persona seed.
3. **The ~0.69 ceiling is the information limit, not the model.** Both 35- and 80-epoch runs peak there. Text is many-to-one with identity → cannot pin the exact held-out person. For a persona creator, this is correct behaviour, not a failure.
4. **Conditioning fix validated:** full-sequence T5 cross-attention + mask-aware pooling delivered the gain; mean-pool could not.
5. **LDA-64 ceiling (from prior ceiling test) = 0.9998** — the representation is near-lossless; all loss was in the Prior's conditioning, now largely closed.

### Harness lessons (logged for "ground it better next time")
- **Save-best-not-last bug:** the script saves the final checkpoint, not the peak. The 80-epoch run peaked at epoch 35 (AUC 0.687) then *overfit* to AUC 0.662 by epoch 80 while train loss fell 0.0017→0.0003. The saved 80-epoch artifact is the overfit one.
- **Attribute gate is epoch-sensitive:** at epoch 80 the negative cosines collapsed (0.15), distorting the margin-based attribute AUC down to skin 0.69 / hair 0.60. The epoch-35 numbers (0.889 / 0.712) are the trustworthy ones. Stop at the verification-AUC peak.
- **Two prior metric bugs (revoked above):** G1 per-dim÷per-image units mismatch (~50× deflation); G2 wrong-space cosine (compressed-vs-compressed). Both are now in the `embedding-evaluation` skill (#13, #14) plus the ceiling-test discipline (#15).

### PREMISE CORRECTION (the most important non-code conclusion of this session)
The original Eidolon premise — that the **50 z_g values would be the identity sliders** — is **dead**. z_g is identity-blind (Fisher J ≈ 0.06); it sculpts pose / yaw / framing / coarse expression only. **Identity sliders, if they exist, must be carved out of the AuraFace-LDA space, not z_g.** The product vision ("Poser with a text seed → sculpt identity with sliders → render via vector-to-image") is unchanged, but the slider substrate moved from z_g to AuraFace-LDA.

### NOT yet established (grounding gaps — drive the next experiment)
1. **Identity generalization to unseen *people* is untested.** All gates ran on FFHQ (1 image/identity), so "held-out" = held-out person but we can never test "different photo, same person." Only the multi-image Hegre corpus can run a true held-out-*persona* (cross-shoot) gate.
2. **Persona-creator viability untested.** Verification-against-exact-person is the wrong objective for the product (we want *a* valid on-description identity, not the exact held-out one). No metric yet measures "coherent, specific, on-description identity."
3. **Trustworthy, disentangled identity-slider directions not yet extracted.** Attribute AUCs prove the directions *exist* (skin 0.889, hair 0.712); the actual semantic-vector extraction (GANSpace/InterFaceGAN-style) is future work.

### Architecture implication for the persona creator
For *generation* the product wants the **stochastic** path (Flow Matching, Arm B): each noise seed → a distinct specific identity within the text-described region. The deterministic regressor (Arm C) predicts the conditional-mean ("average blonde") — higher exact-match AUC but wrong for generating varied specific personas. Arm C's higher AUC is therefore not the selection criterion; it was the cleanest *probe* of where the attribute regions sit.

Status: `[CONCLUDED]` — experiment succeeded; ceiling mapped; premise corrected (z_g≠identity sliders).
Branch: `exp/text-to-zg`.

---

## [PRE-REGISTERED] Phase 5b: Poser Retrieval Spike — text-pin → FFHQ kNN + Hegre cross-shoot gate (`exp/text-to-zg`)

**Date pre-registered:** 2026-06-29
**Branch:** `exp/text-to-zg` (current). May fork `exp/poser-retrieval` if it grows.

### Motivation (closes grounding gaps #1 and #3 from Phase 5a-exp1)
Phase 5a-exp1 concluded text→AuraFace-LDA lands the identity pin in the correct
*attribute region* (skin AUC 0.889, hair 0.712) but left three gaps. Two are
addressable **now, with zero new model training**, using only already-extracted
vectors:
- **Gap #1 (cross-shoot generalization, untested):** every gate to date ran on
  FFHQ (1 image/identity), so "different photo, same person" has never been
  tested. The Hegre corpus is the only substrate that can: live NAS `review.db`
  (`…/eidolon/hegre-faces/v1/review.db`) holds **324 approved personas /
  111,095 approved images**, median **209 imgs/persona**, 321 personas ≥10 imgs.
- **Gap #3 (identity-slider directions not extracted):** attribute AUCs prove the
  LDA directions *exist*; this spike is the cheapest place to extract and
  eyeball them (as **retrieval** sliders) before committing any to the DiT.

### Scope discipline (frozen — read before building)
This is a **retrieval harness + interim UI, NOT the renderer.** "Render" here =
kNN into real FFHQ/Hegre faces, not DiT synthesis. The generative renderer is
Phase 5 (unbuilt). This spike exists to (a) ship a clickable Poser prototype and
(b) de-risk the DiT by validating the pin and the slider directions on real
vectors. Do not let a passing retrieval gate be read as "the generator works."

### Substrate (verified on disk 2026-06-29 — cited, not assumed)
- FFHQ AuraFace: **69,960** per-image `.npy` (`ffhq/auraface/`). 1 img/identity.
- Hegre AuraFace: per-image under `…/hegre-faces/v1/auraface/faces/`; per-persona
  centroids **323** (`averages/*.auraface.npy`) + z_g centroids **324**.
- LDA basis: `experiments/geometry_pca/output/auraface_lda.npz` =
  `lda_basis (512,64)`, `lda_eigenvalues (64,)`, `pooled_mean (512,)`.
  AuraFace preproc = PC1 + yaw removed (`auraface_preprocessing.py`) BEFORE LDA.
- Text→LDA Priors: `output/exp1_g2/exp1_arm_{A,B,C}.pt`. **Arm B (stochastic FM)
  is the product path** (varied specific personas per seed); Arm C (deterministic
  regressor, conditional-mean "average blonde") is a probe only, NOT the product.
- **`hera` orphan MUST be dropped/repaired before indexing** — has a z_g centroid
  but no AuraFace centroid (all frames failed detection).

### Pre-registered gates (FIXED before any eval code; nulls + units explicit)

> **Do-nothing null discipline (logged after 2 revoked verdicts this session):**
> every gate below names its null and its units BEFORE compute. A PASS that does
> not beat its named null is not a PASS.

**G-A (PRIMARY) — Cross-shoot held-out-persona retrieval on Hegre.**
The true generalization test FFHQ cannot give. Hold out K personas entirely
(no images, no centroid in the index). For each held-out persona, take a text
caption of ONE shoot → text→LDA pin → kNN against an index built from the
held-out personas' *other-shoot* images (and/or per-persona centroids computed
from held-out shots only). Metric: **Recall@k** that the nearest neighbour is the
same held-out persona, retrieved from a DIFFERENT shoot.
- **Units:** retrieval is over the SAME LDA-64 space the Prior predicts in;
  compare predicted-pin vs index in that space (NOT raw 512-d, NOT reconstruction).
- **Null (REQUIRED):** random-projection null — replace the trained Prior pin
  with a random Gaussian 64-d vector (matched norm), same kNN. Per the Phase-3
  lesson, ANY structured shadow can clear a naive bar; the Prior must beat the
  random-projection Recall@k by a margin whose identity-bootstrap CI excludes 0.
- **Second null:** caption-shuffle — pair each persona's pin with a *different*
  persona's caption; Recall@k must collapse to the random-projection floor.
- **PASS:** Recall@10 (Prior) > Recall@10 (random-projection null), CI excl. 0,
  AND caption-shuffle ≈ null. (Absolute Recall@k reported but secondary — text is
  many-to-one, so exact-person recall is expected to be modest; the *lift over null*
  is the claim.)

**G-B (SECONDARY) — Attribute-controllability of LDA retrieval sliders.**
Tests Gap #3: does stepping ±σ along an interpretable LDA axis move the retrieved
face's attribute, vs a null direction?
- **Setup:** start from a pin, step +σ and −σ along LDA1 (visually = skin-tone /
  ethnicity, confirmed Phase 5-prep), re-kNN. Score retrieved-face skin tone
  (proxy: existing skin-tone attribute used in 5a-exp1 attribute gate).
- **Units:** σ = the per-axis std of the LDA-64 *index population* (state which:
  FFHQ vs Hegre — they differ; Hegre LDA spread ≈ FFHQ × ~1.9). Step in index units.
- **Null (REQUIRED):** step the SAME magnitude along a RANDOM unit direction in
  LDA-64; the attribute should NOT move monotonically. The LDA-axis Δattribute
  must beat the random-direction Δattribute, CI excl. 0.
- **PASS:** monotone Δ(skin attribute) along LDA1 with |Δ| significantly above the
  random-direction null. (Report hair/LDA-k too; LDA1 is the registered primary.)

### Eliminated / out-of-scope (do NOT pursue in this spike)
- Predicting richer 512-d identity (LDA-64 ceilings at AUC 0.9998 — settled).
- Swapping T5 / retraining the Prior (use existing `exp1_arm_B.pt`; Arm B is
  product path). Retraining is a *different* experiment.
- DiT synthesis / any generative rendering (Phase 5, unbuilt).
- "Verification-against-exact-person" as a PASS criterion — it is the wrong
  objective (5a-exp1 gap #2); G-A measures cross-shoot recall *lift over null*,
  not exact-person verification grade.

### Architecture under test (the corrected Poser substrate)
| Poser control | Substrate | Source artifact |
|---|---|---|
| Pose / emotion sliders | z_g (50-d, identity-blind, J≈0.06) | `encoder_production.npz` |
| Identity pin (text seed) | text→AuraFace-LDA-64, **Arm B stochastic** | `output/exp1_g2/exp1_arm_B.pt` |
| Identity sliders | ±σ along LDA-64 axes (retrieval, this spike) | `auraface_lda.npz` |
| Render (interim) | kNN into real FFHQ / Hegre faces | NAS auraface dirs |
| Render (final, unbuilt) | DiT `prx-tg`, 2-stream cross-attn | Phase 5 |

### Open decisions deferred to results (not gates)
- K (held-out persona count) and the train/index split for G-A — set from the
  324-persona distribution at build time; record the exact split in results.
- Whether to index per-image or per-persona-centroid for Hegre (likely report both).
- Skin-tone proxy provenance for G-B (reuse 5a-exp1 attribute scorer; cite it).

Status: `[CONCLUDED — G-A FAIL (informative); GT-LDA ceiling PASS (cross-shoot validated)]`.
Branch: `exp/text-to-zg`.


### Verdict — G-A (cross-shoot Prior retrieval): `[FAIL — informative]`

**Date concluded:** 2026-06-30

Run on the corrected Hegre corpus: 2,999 query images (one held-out shoot each),
30,000 index images, **242 personas** with ≥2 T5+AF sets. Prior Arm B (stochastic
FM) with training-faithful T5 masking (t5_mask → valid tokens → cap 256 →
zero-pad; see §Bug found below). Persona-level bootstrap, 2,000 resamples.

| Query source | R@1 | R@5 | R@10 |
|---|---|---|---|
| Prior (masked) | 0.010 | 0.037 | 0.072 |
| Random null | 0.006 | 0.030 | 0.053 |
| Caption-shuffle | 0.008 | 0.033 | 0.056 |
| **GT-LDA ceiling** | **0.842** | **0.922** | **0.941** |

Chance R@10 ≈ 0.042 (242-index-persona kNN).

**Bootstrap Δ(Prior − Null), persona-level:**

| k | Δ | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| 1 | −0.002 | [−0.009, +0.002] | 0.79 |
| 5 | +0.003 | [−0.015, +0.021] | 0.38 |
| 10 | +0.014 | [−0.004, +0.033] | 0.063 |

**Verdict: FAIL.** The Prior's cross-shoot Recall@k lift over a random-projection
null does not achieve statistical significance at the persona level. The 95% CI
includes zero at all k; p=0.063 at k=10 is the closest approach (borderline but
n.s. at α=0.05). The effect is directionally positive at k=5 and k=10 and grew
after fixing a conditioning bug (see below), but the corpus (242 personas) is
underpowered to resolve an effect this small. The Caption-Shuffle (0.056) does
not fully collapse to the null (0.053) — inconsistent with the pre-registered
expectation and another weak signal.

**Honest interpretation.** Text→LDA does add *some* cross-shoot identity signal
above random — the positive Δ at k=5 and k=10, the non-collapse of caption-shuffle,
and the consistent improvement after masking fix all point in this direction. But
the effect is too small to be practically useful: at R@10=0.072, a "blonde woman"
pin narrows the search from 242 personas to ~17 plausible matches. That is
*coarse narrowing*, not identity pinning — consistent with Phase 5a-exp1's
information ceiling (verif AUC 0.687; attribute AUC 0.889).

**The GT-LDA ceiling is the decisive positive finding.** When the query is a
REAL held-out-shoot AuraFace vector (no Prior), cross-shoot retrieval hits
R@1=0.842, R@10=0.941. This is the first empirical proof that **AuraFace-LDA is
a genuine cross-shoot identity carrier** — a result no FFHQ gate could produce
(1 image/identity). Tested across four metric/space variants (Euclidean, cosine,
z-scored, reconstructed-512-cosine) — all statistically tied, all >0.83 R@1.
**The retrieval space is sound. The gap is purely in the Prior**, not the
plumbing.

### 🔴 Bug found during review: T5 padding not masked in `predict_pin`

**Date caught:** 2026-06-30. **Severity:** high — confounded every Prior-based
G-A result computed before the fix.

The Prior's training pipeline (`train_exp1_g2.py` lines 94–105) preprocesses T5:
load `t5_mask.npy` → keep only valid (non-padding) tokens → cap to MAX_TOKENS=256
→ zero-pad. The model (`SeqCrossAttnPool`) has **no internal key-padding mask**;
it relies on the caller to zero-out padding rows before feeding them.

The initial `predict_pin` fed the **raw 512-token T5 sequence** including ~347
non-zero T5 padding embeddings the model never saw in training. The cross-attention
pooling attended over garbage padding tokens, corrupting the conditioning signal.

**Fix:** `predict_pin` now accepts a `mask` argument and replicates training
preprocessing exactly (valid tokens via mask → cap 256 → zero-pad to 256).
Verified by invariant test: `test_predict_pin_applies_mask` — padding with
arbitrary garbage produces the identical pin as padding with zeros. All 8 unit
tests pass. This fix is required for any downstream use of the Prior at inference.

**Effect of the fix** (same corpus snapshot, before vs after):

| | Unmasked | Masked (corrected) |
|---|---|---|
| Prior R@10 | 0.056 | 0.072 (+29%) |
| Δ(Prior−Null) k=10 | +0.002 | +0.014 (7×) |
| P(Δ ≤ 0) | 0.41 | 0.063 |

The fix is real and measurable — signal exists but is small.

### Hegre data coverage snapshot (2026-06-30)

- **35,843 images** with both T5 and AuraFace across **238 personas** (30.6% of approved).
- **217 personas** with ≥2 T5+AF sets — viable for cross-shoot. Highest: flora (2,933 in 77 sets).
- Full NAS scan by `scripts/count_full_data_images.py` confirmed `stratum/faces/` is
  the canonical T5 path (preserves the `faces/` DB prefix, matching AuraFace layout).
- Corpus actively growing: query images grew from 2,986 → 2,999 between runs ~40 min apart.

---

## [CONCLUDED] Sapiens2 Keypoints Study — faithfulness + identity carrier (`exp/sapiens2-keypoints-study`)

**Date:** 2026-07-07. **Question:** Are Sapiens2 pose keypoints (a) faithful (measure, not hallucinate like DWPose) and (b) a stronger identity carrier than the 68-pt DWPose landmarks `z_g` is built from? Motivated by the Sapiens2 3D tangent (single-view pointmaps + pose registration were excellent; pose keypoints looked far better than DWPose).

**Cohort:** 25 personas × 15 cross-shoot images = 375 (4–15 shoots/persona). All with image + DWPose (`pose.npy`) + AuraFace (`.npy`). Full code + results: `experiments/sapiens2_keypoints/`.

**Instrument validation (trust the ruler):** DWPose 2D shape reproduces the documented **0.688** ≈ z_g's 0.69 baseline; label-shuffle chance = **0.49**. (Note: random-projection null is NOT a valid floor for a raw geometric feature — Johnson-Lindenstrauss preserves cosine, so projection ≈ feature. Use label-shuffle.)

### Arm A — Faithfulness: Sapiens2 wins decisively
| Test | Sapiens2 | DWPose |
|------|----------|--------|
| Per-keypoint confidence (seen vs unseen kp) | 0.901 vs 0.275 | no confidence |
| Withholds occluded points | ~22%/img (33% on profiles) | 0% — all 68 always |
| Within-person config scatter (lower=faithful) | **0.0076** | 0.0243 (3× noisier) |

Confirms Tim's intuition: DWPose prioritizes completeness over correctness (plants all 68 regardless of visibility); Sapiens2 expresses genuine per-keypoint uncertainty and its kept points are 3× more stable within-person.

### Arm B — Identity discrimination (cross-shoot verification AUC, 3-seed)
| Feature | AUC |
|---------|-----|
| Label-shuffle chance | 0.49 |
| DWPose 68-kp, 2D shape | 0.688 (=documented z_g) |
| **Sapiens2 kp, 2D shape** | **0.766** (G1 PASS, +0.077) |
| z_g / DWPose 3D-frontalized (documented) | 0.67–0.69 |
| **Sapiens2 kp, 3D-frontalized (measured-GPA)** | **0.734** (G2 PASS) |
| Sapiens2 kp, 3D-frontalized (template-lift, z_g recipe) | 0.736 |
| AuraFace-LDA (ceiling) | 0.998 |

### Confound resolved (method vs density)
Sapiens2 keypoints through z_g's **exact template-lift** recipe = 0.736 vs measured-GPA 0.734 (Δ=−0.002). So the 0.69→0.734 gain over z_g is **100% keypoint-source** (dense faithful landmarks), NOT the frontalization method. Measured 3D depth adds ~0 identity beyond the 2D configuration — echoes the dead z_d path (monocular depth is identity-poor).

### Conclusions
1. **Sapiens2 keypoints > DWPose for identity, method-independent** (+0.077 in 2D, +0.04 in 3D pose-removed). z_g conclusion **refined, not overturned**: landmark shape is a *weak* identity discriminator (0.73 ≪ 0.998); density buys ~+0.04.
2. **NOT an AuraFace replacement** — geometry loses the who-is-this race decisively (consistent with FLAME β=0.585, z_d/z_a dead paths). Identity lives in appearance, not landmark geometry.
3. **Product-relevant reframe:** Sapiens2 dense landmarks/pointmap are a candidate **editable-morphology / sculpting substrate** for the Poser (anatomical sliders + angle control) — a complementary control stream, not the identity carrier. AuraFace-LDA keeps identity.

### Recommendation (actionable)
**Replace DWPose with Sapiens2 pose wherever z_g-style geometry is computed** — same pipeline, +0.04 identity, 3× more stable, honest per-point uncertainty. A z_g rebuilt on Sapiens2 keypoints is strictly better input.

### Caveats / next
- 25 personas; widen to ~100 + persona-level bootstrap CI for a production number.
- Study runs on the standalone Sapiens2 backbone + local 1B checkpoints (see script headers); not yet wired into `tools.hegre_dataset`.

### Widened to 100 personas + persona-level bootstrap CI (2026-07-07)

Production-grade re-run: 100 personas × 15 cross-shoot imgs = 1500. Both gates from 2D keypoints (step-3 proved template-lift ≡ measured-GPA). Bootstrap resamples **personas** (correct for clustered observations), 200 resamples.

| Feature | AUC | 95% CI |
|---------|-----|--------|
| Sapiens2 2D shape | 0.745 | [0.725, 0.763] |
| DWPose 2D shape | 0.650 | [0.633, 0.667] |
| AuraFace-LDA | 0.995 | — |
| **Δ(Sapiens2 − DWPose)** | **+0.096** | **[+0.079, +0.113]** |

**P(Δ≤0) = 0.000** — Sapiens2 reliably beats DWPose. At 4× the cohort the gap *widened* (+0.096 vs +0.077 at n=25). Absolute AUCs are lower than the 25-persona run (harder 100-way task; DWPose 0.650 < documented 0.69 on this low-zg 100-persona set) but the **relative delta is the robust, production number**. Conclusion unchanged and now statistically bankable: replace DWPose→Sapiens2 for z_g input.

### ⚠️ CORRECTION — Fisher-J transient/morphology split: z_g STAYS on DWPose (2026-07-07)

The earlier "replace DWPose→Sapiens2 for z_g" recommendation is **superseded**. Tim raised the key objection: the fidelity that makes Sapiens2 a better identity carrier also breaks z_g's *identity-blindness* (the property disentanglement relies on). Tested via Fisher-J axis split.

**Confound caught first:** initial split on the low-zg cohort gave inflated J (DWPose 0.322 vs documented 0.06 — the tell) because sorting on zg suppresses within-person variance (denominator of J). Re-ran on a mixed-zg cohort (100 personas, zg span ~8.0, capped <15 to exclude DWPose-failure tail).

**Mixed-cohort result:**
| | Sapiens2 | DWPose |
|---|---|---|
| Mean Fisher J | 0.331 | 0.136 (→ documented 0.06) |
| Morphology axes (J>0.15) | 42 | 10 |
| Transient axes (J<0.05) | **0** | **0** |
| min axis J | 0.099 | — |

**No identity-blind transient block exists in Sapiens2 shape** — even with natural pose/expression variance, every axis is person-discriminative (dense keypoints capture identity so well that *how a face moves is itself identity*). You cannot carve an identity-blind z_g out of Sapiens2 by axis selection.

**But** full Sapiens2 shape → AuraFace ridge R² = **−0.11** (more orthogonal than z_g's −0.03). Sapiens2 shape-identity is linearly independent of AuraFace appearance-identity.

**Corrected architecture conclusion (two opposed roles, one representation can't serve both):**
- **z_g = identity-blind pose control → KEEP DWPose.** Its coarseness IS the identity-blindness (J→0.06); load-bearing. Do NOT drop-in replace with Sapiens2.
- **Sapiens2 = NEW complementary shape/morphology stream** (AuraFace-orthogonal, editable sliders + angle) — a third conditioning axis alongside DWPose→z_g (pose) and AuraFace-LDA (appearance), NOT a z_g upgrade. Guard non-linear leakage with CFG dropout.

Net: the Poser gains three orthogonal-ish handles (pose / appearance-identity / shape-morphology) — the substrate the "pick identity + sculpt nose/eyes + change angle" product needs.

---

## [PRE-REGISTERED] Reproducibility Sprint: LDA Basis Refit with Cleaned Hegre Dataset (`exp/geometry-pca`)

**Date:** 2026-07-20
**Status:** `[CONCLUDED — G1 PASS (basis quality); G2 PASS (retrieval ceiling)]`

### Goal

Refit the AuraFace-LDA basis (`auraface_preprocess.npz` + `auraface_lda.npz`) on the cleaned Hegre dataset (324 personas, 166k approved, non-faces purged). The dataset underwent massive cleaning: 97k unreviewed → classified, ~54k bad_geometry → extraction_nonface, ~18k contamination purged. The previous basis was fitted on 151 personas / 55,680 images (2026-07-03). Re-fit with 2.1× more personas and 3× more images to improve cross-shoot retrieval ceiling.

**Governance:** `experiments/geometry_pca/provenance_refit_cleaned.yaml` + `config_refit_cleaned.yaml`

### Pre-registered gates

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| G1 (Basis quality) | intra-class scatter ≤ old basis, inter/intra ratio ≥ baseline | Qualitative |
| G2 (GT-LDA ceiling) | cross-shoot R@1 ≥ 0.842 | Target ≥ 0.85 |
| G3 (Corpus integrity) | build-corpus ≥ 31,668 samples | —

### Infrastructure repairs

During execution, two modules referenced by the CLI were discovered missing from git (never committed during the 2026-07-03 session):
- `tools/hegre_dataset/review/fit_lda_basis.py` — the fitting code (PCA + LDA)
- `compute_lda_vectors()` in `tools/hegre_dataset/review/geometry.py` — per-persona averaging
- `data_loader.py` and `corpus_builder.py` still used raw `sqlite3.connect("review.db")` — migrated to `HegreDataset` (PostgreSQL)

All were implemented from scratch and committed.

### Empirical Evidence: G1 — Basis quality

**Data:** FFHQ 69,960 + Hegre 166,195 = 236,155 pooled vectors; 324 personas, 166,195 approved Hegre images.

| Metric | Old (2026-07-03) | New (2026-07-20) | Δ |
|--------|-------------------|-------------------|-----|
| Hegre vectors | 55,680 | **166,195** | 3.0× |
| Personas | 151 | **324** | 2.1× |
| Pooled total | 125,640 | **236,155** | 1.9× |
| PC1 variance | 2.05% | 1.99% | slightly tighter |
| LDA train personas | ~121 | **259** | 2.1× |
| LDA train images | 41,274 | **137,102** | 3.3× |
| LDA eigenvalues | 0.007–0.060 | 0.005–0.032 | more compressed |
| Per-persona averages | 324 | **324** | all L2-norm=1.0 |

**Yaw direction preserved** from existing basis — recomputing requires 166k `pose.npy` loads over NAS (prohibitively slow). The yaw direction (head-pose cleanup, R²=0.54) is stable across dataset changes. PC1·yaw orthogonality drifted to 0.016 (was ~0.000) due to shifted PC1 axis — acceptable for identity discrimination.

### Empirical Evidence: G2 — GT-LDA retrieval ceiling

Cross-shoot evaluation: query = held-out-shoot AuraFace → LDA, index = remaining shoots. 11,110 query images across 321 personas, 30,000 index (sampled). Chance R@10 = 3.1%.

| Variant | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| **A. GT-LDA64 Euclidean** | **0.8538** | 0.9305 | 0.9507 |
| B. GT-LDA64 cosine | 0.8504 | 0.9307 | 0.9528 |
| C. GT-LDA64 z-scored | 0.8484 | 0.9272 | 0.9475 |
| D. GT recon→512→L2norm | 0.8134 | 0.9001 | 0.9249 |

**G2 PASS: R@1 = 0.8538 > 0.842 (old ceiling) > 0.85 (target).** +1.2 pp improvement from the cleaned dataset and refitted basis. Euclidean (A) beats cosine (B) by 0.003 at R@1 — the original metric space remains best. Z-scoring (C) slightly hurts. Reconstruction round-trip (D) loses information.

### G3 — Corpus integrity

**PASS.** Full rebuild executed on the cleaned dataset: **31,711 samples / 321 personas, 0 errors** (target ≥ 31,668). On-disk sample dirs match `_manifest.json` exactly (0 orphans, 0 incomplete in a 500-sample audit); basis fingerprint `e2f66241288e1f50`.

**The pre-rebuild corpus was a mixed-basis generation.** The old corpus held **37,011 dirs**, not the ~31.7k a single build produces. Because samples are named `{persona}--{stem}` ("stable across rebuilds") and every sample stores the **persona average** (not per-image vector) as `auraface_lda.npy`, dirs that a later build does not re-select simply survive — carrying the *previous* basis. Diff of old vs new sample sets:

| | count |
|---|---|
| old corpus dirs (pre-rebuild) | 37,011 |
| new corpus (manifest) | 31,711 |
| **stale dirs** (in old, not re-selected) | **5,306** (14.3%) |
| new dirs (not in old) | 6 |

The 5,306 stale dirs carried pre-refit persona averages → training on the old corpus would silently mix bases. Old corpus preserved at `hegre_corpus.old`.

Two operational findings recorded:
- The first build was **killed by a session interruption at 81.4%** (25,820/31,711) — agent-spawned background processes do not survive session restarts. Fixed structurally: added `--skip-existing` (idempotent, resumable) + a `_manifest.json` recording the basis fingerprint and full sample list. Re-run: 25,815 skipped + 5,896 written = 31,711 in 19.4 min, 0 errors.
- **4 orphan dirs** appeared where images were re-labelled `tainted:extraction_nonface` *between* the two runs (the review UI's DONE spawns a background `compute-geometry` job that can keep writing taint labels minutes after it returns). Quarantined to `hegre_corpus.orphans-20260923/`.

### Adversarial pass

- [x] Metric code (retrieval harness) tested — Phase 5b scripts validated 2026-06-30
- [x] Metric definition unchanged — same R@k cdist-based recall
- [x] Result reproduced — 4 variants consistent, Euclidean dominant
- [x] Extremes inspected — ceiling rise (+1.2pp) directionally consistent with cleaner data
- [x] G3 corpus integrity cross-checked against the manifest AND a raw on-disk diff (37,011 vs 31,711) — the 5,306 stale dirs were *measured*, not assumed; the 4 orphans were traced to a label change between runs rather than waved away
**Verdict: PASS (G1 + G2 + G3)**

### Verdict

**GO** — The refitted basis on the cleaned 324-persona dataset raises the GT-LDA ceiling from R@1=0.842 → **0.854**. The retrieval space is sound; the identity conditioning target for Phase 5 (DiT fusion) is now 0.854. The gap between Prior (R@10=0.072) and ceiling (R@1=0.854) remains the core challenge.

### Artifacts

- `experiments/geometry_pca/output/auraface_preprocess.npz` — refitted (backup at `.npz.bak-20260720`)
- `experiments/geometry_pca/output/auraface_lda.npz` — refitted (backup at `.npz.bak-20260720`)
- `averages/*.lda.npy` — 324 per-persona L2-normalized averages
- `experiments/geometry_pca/output/phase5b_gt_lda_refit_20260720.json` — retrieval results
- `experiments/geometry_pca/provenance_refit_cleaned.yaml` — governance
- `experiments/geometry_pca/config_refit_cleaned.yaml` — canonical parameters
- `tools/hegre_dataset/review/fit_lda_basis.py` — NEW (was missing)
- `geometry.py` — `compute_lda_vectors()` added
- `corpus_builder.py` — migrated to HegreDataset (PG)
- `data_loader.py` — migrated to HegreDataset (PG)
- `experiments/geometry_pca/scripts/46b_phase5b_gt_lda_refit.py` — fast-path retrieval script
- `hegre_corpus/_manifest.json` — corpus manifest (basis fingerprint `e2f66241288e1f50`, 31,711 samples)
- `hegre_corpus.old` — pre-rebuild corpus (37,011 dirs, mixed-basis) — retained for audit
- `hegre_corpus.orphans-20260923/` — 4 dirs whose images were re-labelled tainted between runs

### Pending

- Sapiens2 widen to 100 personas (already planned, cleaner data simplifies cohort selection)
- Optionally purge `hegre_corpus.old` (233 GB) once the new corpus has been exercised

---

## [PRE-REGISTERED] FFHQ Basis Reprojection — pre-refit identity stream (`exp/sapiens2-keypoints-study`)

**Date:** 2026-09-23
**Arm:** `ffhq-basis-reproject` (`experiments/ffhq_basis_reproject/`)
**Goal:** Repair `ffhq/stratum/{id}/auraface_lda.npy`, which is on the **pre-refit** LDA basis, so the FFHQ identity stream is encoding-identical to `hegre_corpus`.

### Background — why this exists

`refit-cleaned` refit the pooled LDA basis (2026-07-23) and rebuilt the hegre corpus, but **never reprojected FFHQ**. `prx-tg/production/data_stratum.py` loads the identity vector directly:

```python
identity_emb = np.load(d / 'auraface_lda.npy')   # (64,) float64
```

with no basis check. Every Eidolon-adapter arm whose `stratum_dirs` included `ffhq/stratum` (weight 2.3) therefore trained a 64-d identity slot receiving **two incompatible encodings**. Measured:

| | ffhq/stratum | hegre_corpus |
|---|---|---|
| basis | pre-refit | refit |
| norm | 0.350 (σ 0.035) | **exactly 1.000000** |
| semantic level | per-image | per-persona |

Bit-exact confirmation that the stored files are pre-refit: `‖stored − project_old(raw)‖ = 0` on 6/6 samples, `‖stored − project_new(raw)‖ ≈ 153`.

**Consequence:** the identity conditioning of the five `exp/eidolon-conditioning` arms is confounded. No identity-binding conclusion can be drawn from any of them, including Arm O's PASS (whose geometry result stands independently).

### Premise audit — run 2026-09-23 BEFORE this gate was formalised (recorded honestly)

`scripts/reproject_lda_ffhq.py --dry-run`, 250 samples:

| check | result |
|---|---|
| stratum dirs (real) | 70,000 (+1 `@eaDir`) |
| with raw AuraFace / with auraface_lda | 69,960 / 69,960 |
| raw-but-no-lda (creatable) / lda-but-no-raw (unfixable) | 0 / 0 |
| `‖stored − project_old(raw)‖` | mean **0.00000000**, max **0.00000000** |
| `‖stored − project_new(raw)‖` | mean **153.5432**, max 157.8826 |
| basis fingerprint | `120e1c5a1dc4f423` |

**Premise CONFIRMED.** This was observed before the gate text below was written; that ordering is disclosed rather than concealed.

### Pre-registered gate (stated BEFORE the run)

> **G1 (PREMISE, already observed above):** on ≥250 pre-run samples, max `‖stored − project_old(raw)‖ = 0` AND mean `‖stored − project_new(raw)‖ > 100`. **Observed: 0.00000000 / 153.5432 → PASS.**
> **G2 (CORRECTNESS):** on ≥300 post-run samples, `‖stored − project_new(raw)‖ < 1e-9` AND `|L2norm(stored) − 1.0| < 1e-6`. Any violation → FAIL.
> **G3 (COVERAGE + INTEGRITY):** exactly 69,960 targets written, 0 errors; backup archive contains 69,960 entries; `BASIS_FINGERPRINT.json` stamped. Any mismatch → FAIL.
> **G4 (GUARD EFFICACY — negative control):** the guard must be able to FAIL. `assert_basis` raises `BasisMismatch` on an unstamped dir and on a dir stamped against a different basis. A guard that cannot fail does not count.
>
> **FAIL if** any of G2/G3 fails, or the premise fails to reproduce.

### Target convention

**Refit basis + L2-normalize (norm 1.0)** — matches `hegre_corpus`, which is what the DiT consumes. L2 normalization provably does not change cosine geometry (between-image cosine identical before/after, 0.9945 ± 0.0012), so this costs no identity information. Rejected alternative: raw refit coords (norm ≈ 153), the convention of the per-image retrieval tree `hegre-faces/v1/lda/`, which is *not* what the DiT consumes.

### Out of scope (checked, recorded)

- `hegre-faces/v1/lda/` holds **2,220 stale pre-refit files** (0.75% of 295,468). All 2,220 are `tainted:extraction_nonface` (2,219) or `tainted:contamination` (1) — **none approved**, so none reachable from training, the corpus, or the GT-LDA ceiling. Left untouched.
- FFHQ's 139 missing `z_g.npy` / 41 missing `auraface_lda.npy` — do not intersect the 69,960-file reprojection surface.
- **Reprojection fixes the basis, not the semantics.** FFHQ is ~1 image per identity, so even corrected its identity vectors still teach "identity vector = per-image key". It becomes encoding-consistent, not a valid identity target.

### Results (run 2026-09-24)

`scripts/reproject_lda_ffhq.py --apply --force`, 536 s, CPU-only:

| metric | value |
|---|---|
| targets | 69,960 |
| written | **69,960** |
| errors | **0** |
| throughput | ~131 files/s |
| basis fingerprint | `120e1c5a1dc4f423` |
| HEAD at run start | `b35a3e6` (working tree dirty — the arm's scripts were uncommitted at run time) |

Backup: `_auraface_lda.oldbasis-backup.tar.gz`, 36.5 MB, **69,960 entries** — the pre-refit files are preserved, not destroyed.

Stamps written for all three consumed dirs: `ffhq/stratum`, `hegre_corpus` (refit basis + L2-normalize), `hegre-faces/v1/lda` (refit basis, raw coords).

**Incident, recorded:** the first `--apply` attempt failed on all 27,475 files it reached. Root cause: `np.save("…/auraface_lda.npy.tmp", v)` — numpy **appends** `.npy` when the path lacks that suffix, creating `auraface_lda.npy.tmp.npy`, so the subsequent `os.replace` found no source. No data was modified (the replace never ran) and the backup was already complete, so the failure was fully recoverable. Fixed by writing through a file handle (`with open(tmp,'wb') as fh: np.save(fh, v)`); 27,475 stray `.tmp.npy` files removed via a new `--clean-litter` action; re-run clean. **Lesson: never hand a non-`.npy` path to `np.save` in an atomic-write pattern.**

### Adversarial pass

- [x] **Metric code tested** — `tests/tools/test_basis_fingerprint.py` (7 tests, incl. `test_different_basis_is_detected` as a negative control). The apply path and the verify path are *independent* implementations (verify recomputes from raw), and the adversarial audit below used a third path.
- [x] **Metric definition stable** — basis fingerprint `120e1c5a1dc4f423` recorded in both the manifest and the stamps; projection convention recorded per directory.
- [x] **Result reproduced** — recomputation from raw on a **random** 250-file sample (deliberately not the first N, which `--verify` had sampled) in a fresh process: **0 mismatches**.
- [x] **Extremes inspected** — **FULL scan of all 70,000 dirs**, not a sample: 69,960 at unit norm, 0 with norm ≠ 1, 0 near-zero/degenerate, 0 NaN/Inf, 0 wrong-shape, 40 missing (the known gap). `norm min = max = mean = 1.000000000` exactly. Single mtime day (all rewritten).

```
Verdict: PASS (G1 + G2 + G3 + G4)
```

### Verdict

**GO** — the FFHQ identity stream is now encoding-identical to `hegre_corpus` (both norm `1.000000000`, both on basis `120e1c5a1dc4f423`). The mixed-basis confound that silently affected every `eidolon`-adapter arm with FFHQ in its `stratum_dirs` is removed, and a guard now exists so it cannot recur silently.

**What this does NOT fix:** FFHQ remains ~1 image per identity, so its identity vectors still teach "identity vector = per-image key". FFHQ is now *encoding-consistent*, not *suitable as an identity target*.

### Artifacts

- `ffhq/stratum/_auraface_lda.oldbasis-backup.tar.gz` — pre-refit files (69,960 entries, 36.5 MB)
- `ffhq/stratum/_auraface_lda.reproject_manifest.json` — counts, fingerprint, git commit
- `ffhq/stratum/BASIS_FINGERPRINT.json` + 2 more stamps
- `scripts/reproject_lda_ffhq.py`, `tools/hegre_dataset/basis_fingerprint.py`, `tests/tools/test_basis_fingerprint.py`

---

## [CONCLUDED — KILL of the belief] z_g Validity Threshold — is the high-norm tail detector failure or genuine pose? (`exp/zg-validity`)

**Date:** 2026-09-24
**Status:** **CONCLUDED — KILL (of the belief).** The claim *"`z_g` norm > 25 =
degenerate"* is **falsified**. See Results below.
**Mode:** confirmatory (`provenance.yaml`) — gate locked before any measurement.

> ⚠️ **Instrument defect found and fixed mid-run (recorded, not hidden).** The first
> G1 render applied a **vertical mirror** to the pose points
> (`py = (1-(y+1)/2)*H`; the writer uses no y-flip). That render was invalid and its
> visual read was discarded. `G2` was **unaffected** — it reads raw pose coords and
> never applies the pixel mapping. The fix is in
> `experiments/zg_validity/src/run_gates.py::_denormalize`, with the writer's
> convention quoted inline. The defect was caught by the user, not by my own
> "are the points centred?" check — **a mirror preserves the centroid**, so that
> check was blind to it. Orientation is now verified by anatomical ordering
> (`brow < eye < nose < mouth < jaw`), which is mirror-sensitive.

**Goal:** Decide whether per-image `z_g` vectors in `hegre_corpus` with norm > 25 are
DWPose/encoder failures to be filtered, or genuine pose/expression extremity that must
be kept — and if a filter is justified, specify a defensible criterion.

**Why it matters:** `z_g` is the geometry control, consumed **per-image** by the DiT.
~6.7% of the corpus sits past the value the project's own extraction code calls
degenerate. If those vectors are garbage, the geometry control trains on garbage; if
they are real signal, filtering deletes real pose coverage. **The two readings imply
opposite actions.**

**Prior art (process steps 2–3, checked):** NO concluded arm and NO ledger
adjudication of `z_g` validity exists. Two conflicting thresholds do exist in the
codebase, **neither applied to shipped per-image data**:

| threshold | where | scope it is applied to |
|---|---|---|
| `> 25` | `extract_zg_and_averages.py` L130–131 — *"Reject degenerate z_g (DWPose missed eyes/face → wild PCA projection)"* | **persona averages only** |
| `< 15` | 2026-07-07 Fisher-J cohort cap — *"to exclude DWPose-failure tail"* | one analysis cohort |

**Mechanism (why the norm is ambiguous):** `encode_zg` is
`frontalize → center_and_scale → align_single → PCA(50) → whiten`, i.e.
`z_g = (raw − whiten_mu) / whiten_sigma`. Whitening divides each component by its own
std, **amplifying the low-variance high-index components**. A large norm therefore
means "large projection along a low-variance axis" — which is equally consistent with
(a) a garbage shape from failed keypoints, and (b) a genuine extreme pose that
frontalization failed to remove. The ledger's 2026-07-07 `z_g → yaw R² = 0.98` makes
(b) live: `z_g` demonstrably carries pose. **The discriminator is the keypoints, not
the norm.**

### Pre-registered gate (stated before results)

- **G0 — instrument trust (abort if it fails).** `hegre_corpus` `z_g.npy` bit-identical
  to the `zg/` source tree on this arm's sample. Previously ‖diff‖ = 0.00000000 over 600
  matched samples; re-confirm. If it fails, all downstream conclusions are about a copy.
- **G1 — mechanism, visual (the core question).** 60 images at norm > 25 vs 60 controls
  at norm 8–12; DWPose 68 keypoints rendered over the pixel; each classified plausible
  vs implausible. **Decision rule fixed now:**
  - ≥70% of high-norm **implausible** → **H1** detector failure → a filter is justified
  - ≥70% of high-norm **plausible** → **H2** genuine extremity → **no filter**
  - in between → **MIXED** → criterion must be keypoint-based, not norm-based
  Visual verification is mandatory (standing rule; Phase 2b `z_a` precedent).
- **G2 — quantitative corroboration, independent of the visual.** High-norm vs control
  on per-keypoint confidence, missing-point count, inter-ocular distance (normalized
  units), keypoint-bbox scale/aspect. Direction test: worse keypoint quality → H1;
  equal quality with larger pose amplitude → H2. **Effect sizes with CIs, not bare
  significance** — a large-n test on a 6.7% subgroup is how a trivial difference gets
  promoted to a finding.
- **G3 — falsify the standing claim.** If H2 holds, *"norm > 25 = degenerate"* is
  falsified; retire it **and** re-examine the persona-average filter it drives, since
  persona averages may have been computed over a biased subsample.
- **G4 — the decision.** Any criterion must be (1) computable per-image from data
  available **at inference time** (else train/inference mismatch — worse than the bug
  it fixes); (2) applied **identically to all splits** (else a split confound); (3)
  justified by a **falsifiable measurement, not a round number**. The `<15` vs `>25`
  discrepancy must be reconciled; **that neither is right stays on the table.**
- **G5 — if the corpus changes.** Governed as a **data update**: backup before first
  write, never overwrite an existing backup, idempotent/resumable, stamped, manifest
  updated, count delta reported, **split re-hashed** (a filter changes membership).

**Falsified if:** G0 fails; or G1/G2 support H2; or no criterion satisfies G4.1.
**KILL condition:** high-norm vectors are valid with valid keypoints and carry genuine
pose signal → change nothing, retire the claim, write `DISCONTINUATION_NOTICE.md`.
**A KILL here is a valuable result** — it removes a wrong belief and simplifies the
architecture.

**Adversarial pass (mandatory before any PASS):** all four boxes, with the G1 contact
sheets preserved under `docs/assets/exp/zg-validity/` as the visual evidence.

**Arm dir:** `experiments/zg_validity/` — `provenance.yaml`, `config.yaml`, `README.md`.

**Expected cost:** CPU-only, minutes. No GPU, no training, no model.

---

### Results (run 2026-09-24, CPU-only, no GPU, no model)

**G0 — instrument trust: PASS.** 400 random corpus samples compared against
`zg/faces/{persona}/{set}/{image_id}.npy`: **0 mismatches, 0 missing, max ‖diff‖ =
0.0000000000.** The corpus is a faithful copy of the encoder output, so every
downstream conclusion is about the encoder, not a copy.

**Sampling (frozen to `src/selection.json`, seed 20260924).** 31,711 samples
scanned: p50 = 8.49, p95 = 27.36, p99 = 42.57; > 15 = 20.08%, > 25 = 6.66% —
**identical to the independent 2026-09-23 audit**, so the corpus has not moved.
Eligible: 2,113 at norm > 25, 8,848 at 8–12. Drew 60 (norms 62.8 → 25.2) and 60
controls (11.9 → 8.0).

**Cross-tab against the review DB (this reframed the arm).** Every corpus sample —
31,711/31,711, and 60/60 in *both* strata — is `approved`. The 215,914
`tainted:extraction_nonface` in the review DB never entered the corpus. So the
stratum is **not** a duplicate of the human reject pile: these are faces the
reviewer passed. That is what makes the high-norm tail a real finding, and it
**already disposes of the docstring's rationale** — "DWPose missed eyes/face" is
handled upstream by review.

**G2 — quantitative corroboration.**

| metric | high (norm > 25) | control (8–12) | reading |
|---|---|---|---|
| `conf_lt_03` | **0** | **0** | no low-confidence keypoints at all |
| `zero_xy` | **0** | **0** | no missing keypoints at all |
| `conf_mean` | 0.787 | 0.948 | mildly lower, nowhere near a filter threshold |
| `iod_norm` | 0.209 | 0.295 | eyes closer together relative to face box |
| `eye_mouth_ratio` | **2.695** | **1.173** | control value is textbook-correct; high is 2.3× |
| `align_residual` | 0.434 | 0.253 | shape deviation 1.7× larger |
| `roll_deg` | 9.06 | −4.84 | more roll |

The docstring's stated mechanism — *"DWPose missed eyes/face → wild PCA
projection"* — **did not happen**: zero missing points and zero sub-0.3 confidence
in either group, with confidence still at 0.787. The signal is **geometric, on
correctly-detected points.**

**G1 — mechanism, visual (the core question): → H2.** 60 high + 60 control
rendered as 68-pt skeleton over pixel (`docs/assets/exp/zg-validity/g1_*.jpg`;
`g1_zoom_{high,ctrl}{4,12}.jpg` face-cropped large panels). Reviewer classification:
- **high stratum** — landmark alignment **accurate**; poses **atypical**, often
  **head-inverted**. "The vast majority gives a pretty accurate information about
  the alignment, pose, gaze of the face."
- **control stratum** — "'perfect': point alignment is strong, but poses are normal."

Accurate alignment in ≥70% of the high stratum → **H2: genuine extremity → no
norm-based filter.** The two strata are separated by **pose atypicality**, not by
landmark quality.

**Verdict: KILL of the belief.** The KILL condition pre-registered for this arm is
met verbatim — *"high-norm vectors are valid with valid keypoints and carry genuine
pose signal → change nothing, retire the claim, write `DISCONTINUATION_NOTICE.md`."*
`z_g` norm is a **pose-atypicality detector, not a degeneracy detector.** Filtering
it would delete real, correctly-encoded pose coverage from the geometry control —
the opposite of what the docstring intends.

**G3 applies (triggered by H2).** The claim *"norm > 25 = degenerate"* is retired,
**and** the persona-average filter it drives must be re-examined: persona averages
were computed over a subsample selected by a criterion now known to be invalid, so
they may be biased. **Not yet done — carried forward.**

**Residual open hypothesis (surfaced by the reviewer, NOT adjudicated here).**
The 2D landmarks are accurate, but the reviewer's mechanism proposal is that these
poses sit **at or beyond the bounds of the Sapiens pose model's training
distribution**, so the *encoding* may be unreliable even when the landmarks are
right. That is a **third** hypothesis (H3) and this arm's gate does not test it —
it would need its own pre-registered arm. It matters because H2 and H3 imply
different actions: H2 = keep everything; H3 = the vectors are real poses but
out-of-distribution for the encoder, which is a train/inference question, not a
corpus-cleaning question.

### Adversarial pass (all 7 boxes)

| box | outcome |
|---|---|
| **0. Null computed** | n/a in the randomised sense; the **control stratum (8–12) is the null** — same pipeline, same review status, same rendering, differing only in norm. |
| **1. Metric tested** | G2's separation is on **distance-based** quantities (`eye_mouth_ratio`, `iod_norm`) that are invariant to the mirroring defect, so the fix could not manufacture the result. Verified: the defect lived only in `_panel`; `face_metrics` never applies the pixel mapping. |
| **2. Metric stable** | Sampling is seeded and frozen; a 600-sample bucket probe shows the corpus is **uniformly 1024×1024**, so the bucket normalisation is not a confound. Norm distribution reproduced the prior audit exactly. |
| **3. Result reproducible** | `src/run_gates.py {select,g0,g2,g1,zoom,review}` — every number in this entry is regenerable from the committed script + frozen `selection.json`. |
| **4. Extremes inspected** | **Yes** — this entire arm *is* the extreme-tail inspection. `g1_zoom_high4.jpg` (norms 62.8/47.5/46.6/45.5) is the explicit top-of-tail artifact. |
| **5. Headline number traced to an exact artifact** | `eye_mouth_ratio` 2.695 vs 1.173 → `docs/assets/exp/zg-validity/g2_diagnostics.json`; review cross-tab → `review_crosstab.json`; visual → `g1_skeleton_{high,ctrl}.jpg`. |
| **6. Every flaw found is FIXED or explicitly gated-not-fixed** | **FIXED:** the mirrored renderer (`_denormalize`, commit `c0ed696`); the stale scratch-path evidence convention. **GATED-NOT-FIXED:** G3's persona-average re-examination; the H3 out-of-distribution question; and the `tainted:approved_bad_geometry` artifact below. **WITHDRAWN as my own errors:** the centroid "alignment verified" check, the eye-darkness test (ran on mirrored coords), and two vision reads of mirrored images. |

### Separate defect found in passing (not this arm's subject)

All 19 `tainted:approved_bad_geometry` rows are from a single persona's shoots
(`alexandra-and-ombeline-*`) and carry a **bit-identical** `zg_distance` of
`1.605683246452827e-05`. Nineteen files sharing the same float to 16 digits is a
failed or placeholder computation, not a measurement — that code path can write a
constant. Also: `zg_distance` does **not** separate faces from non-faces (approved
mean 277.8 vs non-face 276.8), so it is not usable as a validity filter. Both are
out of scope here and **carried forward.**

### Carried forward

1. **G3** — re-examine the persona-average filter driven by the retired claim.
2. **H3** — is the `z_g` *encoding* reliable at pose-distribution extremes? Needs its own gate.
3. **Corpus-quality arm** — extreme-pose and off-frame crops pass review and affect **all three streams** (identity, pose, shape), not just `z_g`. This is the larger finding and the one that protects the prx-tg handoff.
4. **`approved_bad_geometry` constant-distance artifact** — separate defect.

**Evidence:** `docs/assets/exp/zg-validity/` — `g1_skeleton_{high,ctrl}.{png,jpg}`,
`g1_zoom_{high,ctrl}{4,12}.jpg`, `g2_diagnostics.json`, `review_crosstab.json`.
**Code:** `experiments/zg_validity/src/run_gates.py` (+ frozen `selection.json`).

---

## `[PRE-REGISTERED]` `zg-identity-blindness` — re-measure z_g's identity content on the curated corpus

**Date pre-registered:** 2026-09-24 · **Branch:** `exp/zg-identity-blindness` ·
**Mode:** `confirmatory` (declared in `provenance.yaml` before run 1) ·
**Cost:** CPU only, no GPU, no model.

### Why

Three independent reasons; any one sufficient.

1. **The belief is load-bearing.** The pose-vs-identity orthogonality design —
   now the frontier of the prx-tg work — assumes `z_g` is a geometry/pose control
   space carrying almost no identity. prx-tg is about to spend real GPU time on
   that premise.
2. **The number's producing script does not exist.** Verified 2026-09-24 across
   **every branch**: the only consumers of `geometry_pca.fisher.fisher_ratios`
   are `07_gate_sweep.py`, `21_zd_gate.py`, `22_zd_complementarity_diagnostic.py`
   and `32_phase3_systematic_review.py` — all operating on the **legacy 1,448 /
   101** set. The sapiens2 scripts compute a *different* quantity (mean of
   per-axis J: 0.136 DWPose / 0.331 Sapiens2). By this project's own rule — *"a
   ledger number whose producing script no longer exists is not evidence"* —
   **J = 0.059 (below) is not currently evidence.**
3. **The data changed underneath it.** J = 0.059 was measured on the
   **pre-curation** corpus (69,110 samples / 323 personas) while 62k were
   `bad_geometry` and 216k unreviewed, and the entry that recorded it flags it
   *"directional, not final"*. The corpus is now **31,711 / 321, 100% `approved`**.

### Hypotheses

* **H₀:** `z_g` carries almost no identity — global Fisher J ≪ 1, small
  morphology block (J > 0.15).
* **H₁ (declared in advance):** J = S_B / S_W, and DWPose noise inflates the
  denominator S_W. Curation shrinks S_W and therefore **RAISES J.** A materially
  higher J means `z_g` carries more identity than believed and the orthogonality
  premise is weaker than the design assumes. **An increase is the predicted
  outcome, not an anomaly.**

### Instrument

**Global** Fisher J = S_B / S_W via `geometry_pca.fisher.fisher_ratios` — the same
function the legacy gate used, rebuilt in-repo because the original is lost.
Input: per-image `z_g.npy` (50-d), labels = `persona` from the corpus
`metadata.json`. S_B and S_W are always reported separately (a high J via a
collapsed S_B is not identity separability).

**Because the original's "Tier 0.3" filter is lost with its script, the quantity
is not reproducible verbatim.** The arm therefore re-measures a declared **venue
family** — mean face-keypoint confidence floors **0.0 / 0.3 / 0.5**, venue B (0.3)
primary — and reports the spread. Reporting only the venue that agrees with 0.059
would be p-hacking by filter.

### Pre-registered gates (verbatim, written before the first run)

| gate | requirement | falsifier |
|---|---|---|
| **G0** instrument identity | corpus `z_g` bit-identical to encoder source, 400 samples, `max‖diff‖ = 0`, `missing = 0` | mismatch → STOP; input is not the real `z_g` |
| **G1** positive control | `J_auraface ≥ 3 × J_zg` | **arm VOID** — a low J is uninterpretable without proving the instrument *can* detect identity separability |
| **G2** headline | **CONFIRM** if `J ∈ [0.02, 0.12]` AND morphology axes ≤ 10; **FALSIFY** if `J ≥ 0.20` OR morphology axes ≥ 20; else **PARTIAL** | both directions stated above |
| **G3** legacy collapse | attempt to reproduce "27 → 6 morphology axes" | unreconstructible → record `UNREPRODUCIBLE`; **no stand-in substituted** |

### Metric calibration (performed before any corpus result was read)

A synthetic validation of `fisher_ratios` was run **before** the corpus
measurement, to confirm the metric measures what is claimed:

| synthetic case | result | expected |
|---|---|---|
| strong identity structure (C=30, N=600) | **J = 26.31** | J ≫ 1 ✅ |
| **pure noise, random labels** | **J = 0.0560** | ≈ (C−1)/(N−C) = 0.0509 ✅ |
| zero within-variance (degenerate) | **J = 0.0000** via the S_W guard | finite, not inf/nan ✅ |

**The noise case is the important one: J = 0.056 for random labels, against an
original claim of J = 0.059.** The metric's floor is `(C−1)/(N−C)` — it depends on
**N and C**, so **raw J is not comparable across corpora of different size**:

| corpus | N | C | floor (C−1)/(N−C) | original J / floor |
|---|---|---|---|---|
| pre-curation (original measurement) | 69,110 | 323 | 0.00468 | 0.059 / 0.0047 = **12.6×** |
| curated (this arm) | 31,711 | 321 | 0.01020 | — |

**The curated corpus has a floor 2.2× higher purely because N fell.** A smaller
corpus mechanically **raises** raw J. So the raw comparison "J rose vs 0.059"
is **confounded** — it would rise even if `z_g`'s true identity content were
identical.

**Disclosure and handling:** the pre-registered gate is on **raw J** and was
**deliberately not changed** after this calibration was understood. `J_null` and
`J/J_null` are reported as **declared supplementary diagnostics**, calibrated
before any corpus result was read, and the cross-corpus comparison is made in
`J/floor` terms. A post-hoc gate change would have forfeited confirmatory status.

### Base deviation (explicit)

`docs/00_GIT_WORKFLOW.md` rule 2 requires branching from `main`. **This arm
cannot.** Verified 2026-09-24: `main` is **220 commits behind** and lacks
`tools/hegre_dataset/models.py` (the corpus loader), `basis_fingerprint.py`,
`scripts/reproject_lda.py` and the build-corpus tooling — a tree from `main`
cannot load the corpus. Per AGENTS.md the deviation is stated rather than
silently branching from an unrelated `exp/*` branch. **Standing blocker for every
future eidolon arm until `main` is merged forward.**

### Results (run 2026-09-24)

**G0 — PASS.** 400 samples checked, 0 missing, `max‖diff‖ = 0.0000000000e+00`. The
corpus `z_g` is bit-identical to the encoder source.

**G1 — FAIL, as pre-registered, and the failure is a DATA property.** The corpus
`auraface_lda` returned `S_W = 0.0000` exactly → J = 0 by the guard.

> **Verified on disk: all 321/321 personas carry a bit-identical `auraface_lda`
> vector across every one of their samples. Worst within-persona
> `max|diff| = 0.0000000000e+00`.**

**The corpus's `auraface_lda.npy` IS the persona centroid** — one vector replicated
across all of a person's images. Fisher J is mathematically undefined on it
(`S_W = 0` **by construction**), so the pre-registered control could not run on the
array it named. This also confirms the centroid-enrichment construction that was
previously only recalled, and means the corpus already implements the identity
conditioning decided for the prx-tg arm.

**G1b — PASS (post-hoc instrument correction, disclosed).** The pre-registered
*intent* — "AuraFace as a stream known to carry identity" — run on the correct
array (per-image LDA, `lda/faces/{persona}/{set}/{image_id}.npy`, 31,704 loaded /
7 missing):

| stream | J | S_B | S_W | morph axes | **J / noise-floor** |
|---|---|---|---|---|---|
| per-image AuraFace-LDA | **2.0137** | 124.36 | 61.75 | 64/64 | **197.5×** |
| `z_g` | **0.0847** | 15.11 | 178.37 | 11/64 | **8.31×** |

Ratio **23.8×** (gate required ≥ 3×). **The instrument detects identity
separability decisively**, so the `z_g` number is interpretable.

**G2 — PARTIAL** (pre-registered gate, unchanged).

| venue | samples | J | J/J_null | morph axes (J>0.15) |
|---|---|---|---|---|
| A (no filter) | 31,711 | 0.0847 | 8.31× | 11 |
| **B (conf ≥ 0.3) — primary** | **31,711** | **0.0847** | **8.31×** | **11** |
| C (conf ≥ 0.5) | 31,676 | 0.0871 | 8.54× | 13 |

Re-standardised `z_g` gives J = 0.1095 / 0.1114 (venues B / C).

J = 0.0847 is **inside** the CONFIRM band [0.02, 0.12], but morphology axes = **11**
**misses the pre-registered ≤ 10 by one axis** → **PARTIAL**. The threshold is **not
moved**; moving it after seeing the count would forfeit confirmatory status.

**Size-corrected comparison to the original.**

| | N | C | floor (C−1)/(N−C) | J | J/floor |
|---|---|---|---|---|---|
| original (lost script) | 69,110 | 323 | 0.00468 | 0.0590 | **12.60×** |
| this run (venue B) | 31,711 | 321 | 0.01019 | 0.0847 | **8.31×** |

**Raw J rose (0.059 → 0.085) but that is arithmetic**: the curated corpus's floor is
**2.18× higher** purely because N fell. **Size-corrected, `z_g`'s relative identity
signal FELL, 12.60× → 8.31×.** The raw rise carries no information about identity
content; H₁'s predicted rise was produced by the corpus shrinking, not by curation
revealing structure.

**The venue family collapsed — a signal about the lost original.** Venue A ≡ venue B
exactly (31,711 both) and venue C removes only 35 samples: **no corpus sample has
mean face-keypoint confidence < 0.3.** A confidence-based "Tier 0.3" filter would
have been a **no-op** on this corpus, so the original's "Tier 0.3" most likely
denoted something else (a norm tier or a per-identity sample-count tier). Recorded
as an inference, not a finding.

**G3 — `UNREPRODUCIBLE`.** The original's subset (which 1,448 / 101; which tier) is
not reconstructible because the producing script is absent from every branch. **No
stand-in was substituted.**

### 7-box adversarial pass

| box | outcome |
|---|---|
| **0. Null computed** | **Yes, and it is quantitative.** Synthetic validation of `fisher_ratios` before any corpus result: pure-noise random labels give **J = 0.0560** vs theoretical `(C−1)/(N−C) = 0.0509`; strong structure gives J = 26.31; zero-within-variance is finite (J = 0.0000) via the `S_W` guard. The corpus null is `J_null = 0.01019`. |
| **1. Metric tested** | The metric was shown to **fire** (J = 26.31 on real structure) and to **not fire** (J ≈ 0 on noise) *before* the corpus was measured. G1b independently proves it fires on a stream known to carry identity (197.5× floor). |
| **2. Metric stable** | Sampling-free: the arm reads the **entire** corpus (31,711/31,711 loaded, 0 missing `z_g`, 0 missing `auraface`, 0 missing pose). Identities with < 2 samples: **0 dropped** — every persona contributes to within-scatter. Re-standardised variants reported as a scale-sensitivity check (§ above). |
| **3. Result reproducible** | `experiments/zg_identity_blindness/src/run_fisher.py {g0,g1,g2,g3,all}` + `output/fisher_metrics.json` regenerates every number in this entry, including the G1b control. |
| **4. Extremes inspected** | **Yes** — the venue family *is* the extremes sweep, and it collapsed (no sample below conf 0.3), which is a substantive finding about the corpus, not a null result. The superseded `z_g` high-norm tail was already inspected in `zg-validity-threshold` (atypical poses, accurate alignment). |
| **5. Headline number traced to an exact artifact** | `J = 0.0847`, `S_B = 15.1058`, `S_W = 178.3654`, morph = 11, `J_null = 0.01019` → `output/fisher_metrics.json` (venue B block). G1b → same file, `g1b.per_image_lda` (J = 2.0137). Centroid verification → 321/321 bit-identical, produced by the G1b loading path and re-verified standalone. |
| **6. Every flaw found is FIXED or explicitly gated-not-fixed** | **FIXED:** the `ZG_TREE` path bug G0 caught (`geometry_pca_data/zg` → `hegre-faces/v1/zg`); the `names` list-vs-array indexing bug (Pyright-caught); the mis-specified G1 control (→ G1b, disclosed, gate not moved). **GATED-NOT-FIXED:** G1 stays recorded as FAILED even though G1b passes — the pre-registered gate was on the named array and is not retro-fitted; G3 `UNREPRODUCIBLE`; the "Tier 0.3" definition remains unknown. **ACCEPTED DEBT:** the arm was written before the ledger entry (rule 3 ordering), though the design was frozen in the arm's own files before the run. |

### Verdict

**PARTIAL — the belief is not falsified, and its magnitude is corrected.**

`z_g` is **not** an identity space (J ≪ 1; **23.8× weaker than per-image AuraFace**
and only **8.31×** its own noise floor), so the design claim "identity lives in
AuraFace, `z_g` is a geometry/pose control space" **stands in direction**. But the
arm does **not** record CONFIRM: the morphology-axis count (11) missed the
pre-registered ≤ 10 by one axis, and size-corrected the relative signal **fell**
(12.60× → 8.31×) rather than holding. Reported as PARTIAL rather than smoothed into
either verdict.

**Consequence for the prx-tg arm:** the J = 0.059 quoted in
`docs/briefings/2026-09-24_prx-tg_persona-vector-visualization-brief.md` §5 is
**superseded** — it has no producing script and it is size-confounded. The number to
carry forward is **J = 0.0847 (raw) / 8.31× floor (size-corrected), venue B, curated
corpus**, alongside the stronger and more useful fact that **per-image AuraFace sits
at 197.5× the floor** — a 24× separation that is the actual quantitative basis for
the orthogonality design.

**Evidence:** `docs/assets/exp/zg-identity-blindness/fisher_metrics.json` (tracked).
The same file is also written to `experiments/zg_identity_blindness/output/` at run
time, but `output/` is **gitignored** (`.gitignore:205`) so that copy is **not**
evidence — cite the `docs/assets/` path.
**Code:** `experiments/zg_identity_blindness/src/run_fisher.py`
