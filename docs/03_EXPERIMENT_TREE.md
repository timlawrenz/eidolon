# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[ACTIVE] Phase 2: Volumetric Encoder z_d (depth)**
  * z_d from depth.npy. Target ~50 components, whitened. (z_a from normal.npy
    deferred until z_d passes — see below.)
  * **Status:** preprocessing + single-pass NAS depth cache BUILT
    (`18_fit_zd_encoders.py`, `19_build_depth_cache_singlepass.py`); encoder fit
    + gate NOT yet run. Full ledger entry: `02_EXPERIMENTS_AND_RESULTS.md` [Phase 2].
  * Memory strategy for dense maps: (1) seg.npy-mask + face-crop + canonical resample,
    (2) aggressive downsample (~64x64) — identity-level volume is low-frequency,
    (3) sklearn IncrementalPCA (.partial_fit streaming batches) as safe default,
    randomized_svd as the fast spike alternative. Depth cache lives on NAS via the
    `data/` symlink (storage rule: cache CPU/GPU labor, but never on local disk).
  * Normal-map caveat (for z_a later): unit normals live on a sphere; raw-component
    PCA is an approximation — may need tangent-space log-mapping. Spike naive first.
  * New nuisance variables to validate against: lighting (normals) + camera distance
    (depth) are the volumetric analog of pose.
  * **Gate (pre-registered):** `J([z_g | z_d]) > J(z_g) × 1.15` — an
    incremental-information test. Runs on the reviewed, **growing** hegre corpus in
    `data/review.db` (snapshot 2026-06-10: 89 contamination-free identities /
    1,524 approved images), NOT the legacy 10-identity set. Re-runnable as the
    review corpus expands.
* **[TBD] Phase 2b: Albedo/Surface Encoder z_a** (gated behind z_d PASS)
  * z_a from normal.npy. Same recipe; tangent-space log-mapping likely required.
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
