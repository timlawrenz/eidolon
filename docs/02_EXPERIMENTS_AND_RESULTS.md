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
