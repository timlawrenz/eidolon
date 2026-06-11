# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Active & Planned
* **[TBD] Phase 4: DiT Fusion Stack**
  * Decoupled cross-attention + block-diagonal ingestion (architecture.md §7.1).

## Concluded
* **[CONCLUDED] Phase 3: DINOv3 Bridge (Premise Validation)** (`exp/geometry-pca`)
  * Phase 3 (R² Premise): `[FAIL]`. DINO cannot faithfully reconstruct the sliders
    (z_a R²=0.385; z_g C6/C11 ≈ 0 — though C6 is plausibly detector noise, J=0.098).
  * Phase 3b (Identity Transfer): technically cleared 0.51 but **UNINFORMATIVE** —
    review control showed raw dinov3_cls = 0.685 and random 50-d DINO projections
    = 0.627 ± 0.006 ≫ bridge Ŷ_a (0.606). Any DINO shadow passes; the bridge
    *degrades* its input. Gate lacked a random-projection null (lesson recorded).
  * **Real finding:** raw dinov3_cls is the strongest identity carrier measured on
    hegre (AUC 0.685 vs z_a 0.562, z_g 0.540) → DINO = identity conditioning;
    E = interpretable decoupled control. Both fast paths are dead; E non-redundant.
* **[CONCLUDED — PASS] Phase 2b: Albedo/Surface Encoder z_a (normals)** (`exp/geometry-pca`)
  * Surface normals successfully add complementary identity signal over z_g.
  * Structural advantage proven: normals natively resist the affine-scale ambiguity
    that killed raw depth. z_a ALONE (AUC 0.56) is a stronger identity carrier than
    geometry alone (0.54) on hegre editorial data.
  * **Selected variant: `xy` (8,192-d)**. Passed gate cleanly (+0.024 ΔAUC vs ε=0.01 bar),
    most compact, and cleanly avoided the 'rot paradox' (where Rᵀ de-rotation injects
    pose into the global mean via visibility bias).
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
