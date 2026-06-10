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
