# Experiments & Results

This ledger documents the empirical findings and definitive conclusions of all experiments in the Eidolon repository. Negative results are recorded here permanently to prevent repeated failures.

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
- **Mandate:** pose-invariance by construction (architecture.md §3.2 updated).
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
`[ACTIVE]` — EPnP spike in progress on `exp/geometry-pca`.

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
   were a perfect identity carrier, the depth/albedo/DINOv3 partitions would be
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

## [Phase 2] Volumetric Encoder z_d (depth) — `[ACTIVE]`

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
- **Depth preprocessing** (`scripts/13_fit_zd_encoders.py`, commit `3a3793a`):
  seg-mask → face-crop → canonical resample, with **3 normalization modes**
  (A / A_prime / C) to be gated against each other.
- **Single-pass NAS depth cache** (`scripts/17_build_depth_cache_singlepass.py`,
  commit `91b527f`): collapses the old 6-NAS-pass design (3 modes × 2 passes)
  into 1, writing `data/depth_cache/ffhq_depth_{A,A_prime,C}.npy` + `ids.json`.
  Storage-rule compliant — `data/` is a symlink to the NAS project folder.
- **z_d encoder fit** scaffolding (`13_fit_zd_encoders.py`) writing to `output/`.

### What is OPEN (the actual work remaining)
1. Decide/gate the depth normalization mode (A vs A_prime vs C).
2. Fit the frozen `z_d` PCA encoder on FFHQ depth (k≈50, whitened).
3. Build a `z_d` gate extractor over the `review.db` approved set (analog of
   `06_extract_hegre_gate.py`, but reading depth + driven by the DB, not the
   legacy `.npz`).
4. **Run the gate**: compute `J(z_g)` baseline and `J([z_g | z_d])`, check ×1.15,
   write a fresh (non-corrupt) machine-readable results artifact.

### Normal-map / z_a note
`z_a` (albedo/surface from `normal.npy`) is **deferred until z_d passes**. Unit
normals live on a sphere, so raw-component PCA is an approximation that may need
tangent-space log-mapping — gate that separately once z_d is proven.

### Verdict
`[PENDING]` — preprocessing + cache built; encoder fit and gate not yet run.
