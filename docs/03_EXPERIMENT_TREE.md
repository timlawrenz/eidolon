# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[ACTIVE] Phase 1-R: Pose-Invariant Geometry Encoder** (`exp/geometry-pca`)
  * Phase 1 PASS revoked: 2D GPA left yaw/pitch in C1/C2, making z_g pose-entangled (disqualified as identity).
  * EPnP spike: estimate head rotation from 68 pts vs canonical 3D template → rotate frontal → reproject → re-run PCA + new pose-invariance probe.
  * *Gate:* synthetic yaw/pitch variants of one identity must yield near-identical z_g; C1 must read as morphology.
  * Escalate to full 3DMM (morphometrics repo) only if the spike falls short.
* **[TBD] Phase 2: Volumetric Encoders**
  * Run randomized SVD pipeline over masked `depth.npy` and `normal.npy` maps. Target 50 components each.
* **[TBD] Phase 3: DINOv3 Bridge**
  * Linear regression of `dinov3_cls` embeddings to the whitened PCA components.
* **[TBD] Phase 4: DiT Fusion Stack**
  * Decoupled cross-attention + block-diagonal ingestion (architecture.md §7.1).

## Concluded
* **[REOPENED] Phase 1: Geometry PCA Encoder** (`exp/geometry-pca`)
  * Validated GPA + PCA yields clean orthogonal sliders, BUT proved C1/C2 = yaw/pitch.
  * Conclusion: 2D GPA cannot factor out 3D pose; encoder was pose-entangled. Superseded by Phase 1-R.
