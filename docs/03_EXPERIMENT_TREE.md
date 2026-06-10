# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[NEXT] Phase 2: Volumetric Encoders** 
  * Run randomized SVD pipeline over masked `depth.npy` and `normal.npy` maps.
  * Target budget: 50 components per modality.
  * *Gate:* Ensure volumetric scree curves match the exponential decay of Phase 1.
* **[TBD] Phase 3: DINOv3 Bridge**
  * Linear regression of `dinov3_cls` embeddings to the whitened PCA components.
* **[TBD] Phase 4: DiT Fusion Stack**
  * Implement the decoupled cross-attention and block-diagonal ingestion (mandated in architecture.md §7.1).

## Concluded
* **[CONCLUDED] Phase 1: Geometry PCA Encoder** (`exp/geometry-pca`)
  * Validated that GPA + PCA on 68 facial points yields highly structured, orthogonal 2D morphological sliders with perspective neatly partitioned into early components.
