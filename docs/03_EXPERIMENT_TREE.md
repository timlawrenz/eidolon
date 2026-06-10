# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[ACTIVE] Phase 2b: Albedo/Surface Encoder z_a (normals)** — THE PIVOT
  * z_a from normal.npy. Same fit-PCA-whiten recipe; tangent-space log-mapping
    likely required (unit normals live on a sphere — gate naive first).
  * **Why now (not depth):** z_d (depth) CONCLUDED as a dead end for identity
    (see `02_EXPERIMENTS_AND_RESULTS.md`). Surface normals describe the *angle*
    of the surface, not absolute distance, so they natively resist the
    affine-scale / camera-distance ambiguity that fundamentally limited raw
    monocular depth. Structurally positioned to carry a cleaner, scale-invariant
    identity signal.
  * **Gate (verification AUC, NOT trace-J):** `AUC([z_g | z_a]) > AUC(z_g) + ε`
    on the hegre verification test. trace-J is banned for concatenated partitions
    (it's a weighted average — blind to complementarity; see ledger metric bug).
  * Reuse: depth cache pattern (`19`), DB-driven extractor (`20`), verification
    AUC instrument (`23`, to be lifted into a `geometry_pca` helper).
* **[TBD] Phase 3: DINOv3 Bridge**
  * Linear regression of dinov3_cls embeddings to the whitened PCA components.
* **[TBD] Phase 4: DiT Fusion Stack**
  * Decoupled cross-attention + block-diagonal ingestion (architecture.md §7.1).

## Concluded
* **[CONCLUDED — FAIL] Phase 2: Depth Encoder z_d** (`exp/geometry-pca`)
  * Depth (64×64, k=50, FFHQ-fit) adds NO complementary identity signal over z_g.
  * Operational proof: verification AUC z_g=0.541 → +z_d = −0.004 (every mode);
    kNN identity acc 4.3% → −0.2%. Depth slightly *dilutes* the weak geometry signal.
  * Metric bug caught + fixed: trace-J = tr(S_B)/tr(S_W) is a weighted average for
    concatenated vectors → blind to complementarity. Gate instrument switched to
    verification AUC. Cross-examined (4 metrics) to rule out a false-fail.
  * Secondary: z_g verification AUC=0.54 quantifies geometry as a weak identity
    carrier on editorial data (operationalizes Phase 1-R J≈0.08).

* **[CONCLUDED] Phase 1-R: Pose-Invariant Geometry Encoder** (`exp/geometry-pca`)
  * Shipped 3D-frontalized geometry encoder (z_scale=1.0) on 69,851 FFHQ faces.
  * Real-image hegre Fisher gate: 3D beats flat 2D GPA on aggregate identity
    separability (S_W 314→246, S_B holds). Clean-C1 narrative died at n=10.
  * Geometry alone is a modest identity carrier (J≈0.08) — empirically motivates
    the multi-partition E vector. Contamination near-miss documented as a warning.
* **[SUPERSEDED] Phase 1: Geometry PCA Encoder**
  * 2D GPA left yaw/pitch in C1/C2 (pose-entangled). Superseded by Phase 1-R.
