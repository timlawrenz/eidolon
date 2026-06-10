# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[NEXT] Phase 2: Volumetric Encoders (z_d, z_a)**
  * z_d from depth.npy, z_a from normal.npy. Target 50 components each.
  * Memory strategy for dense maps: (1) seg.npy-mask + face-crop + canonical resample,
    (2) aggressive downsample (~64x64) — identity-level volume is low-frequency,
    (3) sklearn IncrementalPCA (.partial_fit streaming batches) as safe default,
    randomized_svd as the fast spike alternative.
  * Normal-map caveat: unit normals live on a sphere; raw-component PCA is an
    approximation — may need tangent-space log-mapping. Spike naive first, gate it.
  * New nuisance variables to validate against: lighting (normals) + camera distance
    (depth) are the volumetric analog of pose.
  * *Gate:* reuse the SAME hegre 10-identity Fisher S_B/S_W gate.
* **[TBD] Phase 3: DINOv3 Bridge**
  * Linear regression of dinov3_cls embeddings to the whitened PCA components.
* **[TBD] Phase 4: DiT Fusion Stack**
  * Decoupled cross-attention + block-diagonal ingestion (architecture.md §7.1).

## Concluded
* **[CONCLUDED] Phase 1-R: Pose-Invariant Geometry Encoder** (`exp/geometry-pca`)
  * Shipped 3D-frontalized geometry encoder (z_scale=1.0) on 69,851 FFHQ faces.
  * Real-image hegre Fisher gate: 3D beats flat 2D GPA on aggregate identity
    separability (S_W 314→246, S_B holds). Clean-C1 narrative died at n=10.
  * Geometry alone is a modest identity carrier (J≈0.08) — empirically motivates
    the multi-partition E vector. Contamination near-miss documented as a warning.
* **[SUPERSEDED] Phase 1: Geometry PCA Encoder**
  * 2D GPA left yaw/pitch in C1/C2 (pose-entangled). Superseded by Phase 1-R.
