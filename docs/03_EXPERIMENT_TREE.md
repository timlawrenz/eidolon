# Eidolon Experiment Tree

A living map of ideas, plans, and active workstreams. 
Link directly to the `exp/*` branch where the work lives.

## Settled Conditioning Stack

**Identity:** flesh-masked DINOv3 patch tokens (Phase 4, AUC 0.797, cross-shoot verified)
**Control:** `z_g` — 50-d pose-invariant geometry encoder (Phase 1-R)
**Dead:** z_d (depth), z_a (normals), DINO→slider bridge (Phases 2–3)
**Next:** Phase 5 — DiT fusion stack, 2-stream decoupled cross-attention

---

## Active & Planned
* **[ACTIVE] Phase 5a: Semantic Geometry Regression** (`exp/text-to-zg`)
  * Decouple semantics by training an MLP to map `[T5 || \overline{AuraFace}] -> z_g`.
  * Persona-averaged AuraFace neutralizes pose leakage discovered in Tier 0.2.
  * Requires `caption` and `t5` enrichment on Hegre.
* **[TBD] Phase 5: DiT Fusion Stack** (`exp/geometry-pca`)
  * 2-stream decoupled cross-attention + block-diagonal ingestion (01_VISION_AND_ARCHITECTURE.md §7).
  * Conditioning stack (settled by Phases 2/2b/3/4): flesh-masked DINOv3 patch
    tokens (identity) + z_g expanded tokens (interpretable geometry control).
  * Volumetrics dead. DINO bridge dead. Architecture validated down to 2 streams.

## Concluded
* **[CONCLUDED — PASS] Phase 4: Masked Patch Tokens (Semantic Face Isolation)** (`exp/geometry-pca`)
  * **Opened:** 2026-06-11. **Concluded:** 2026-06-11.
  * Seg-masked (flesh-only) mean-pooled DINOv3 patches beat the cls baseline:
    AUC 0.797 vs 0.769; bootstrap Δ +0.027, 95% CI [+0.014, +0.045].
  * Effect decomposes: patch-pooling +0.014, flesh-scoping +0.015.
  * Cross-shoot-only AUC (leakage removed by construction) reproduces the
    ordering within 0.001 → the lift is pure cross-shoot identity signal.
  * flesh+hair statistically tied (+0.002); flesh-only selected (hair = shoot-
    styled confound). Alignment audited: row-major from idx 5, 0/1,577 mismatches,
    visual patch-PCA proof. Script: `37_dino_patch_face_pooling.py`.
  * Identity conditioning settled. See ledger Phase 4 and 01_VISION_AND_ARCHITECTURE.md §6.
* **[CONCLUDED] Phase 3: DINOv3 Bridge (Premise Validation)** (`exp/geometry-pca`)
  * Phase 3 (R² Premise): `[FAIL]`. DINO cannot faithfully reconstruct the sliders
    (z_a R²=0.385; z_g C6/C11 ≈ 0 — though C6 is plausibly detector noise, J=0.098).
  * Phase 3b (Identity Transfer): technically cleared 0.51 but **UNINFORMATIVE** —
    review control showed raw dinov3_cls (face crop) = 0.766 and random 50-d DINO projections
    = 0.712 ± 0.007 ≫ bridge Ŷ_g (0.704). Any DINO shadow passes; the bridge
    *degrades* its input. Gate lacked a random-projection null (lesson recorded).
  * **Real finding:** raw dinov3_cls (face crops) is the strongest identity carrier
    measured on hegre (AUC 0.766 vs face-crop z_g 0.67–0.69, z_d 0.56) → DINO =
    identity conditioning; E = interpretable decoupled control. Both fast paths are
    dead; E non-redundant. (Caveat: DINO AUC includes same-shoot context — see C5.)
* **[CONCLUDED — FAIL] Phase 2b: Surface Normals Encoder z_a** (`exp/geometry-pca`)
  * **[2026-06-11] Face-Crop Re-run OVERTURNS previous PASS.** Re-tested on the
    `hegre_faces_stratum` dataset using a seg-clean subset (fg≥30%) to prevent
    the seg-collapse trap.
  * Baseline z_g (0.688) → +z_a (0.649) = **ΔAUC −0.039 (FAIL)**.
  * *The counter-intuitive reality:* Visual inspection confirmed Sapiens depth/normals
    on these crops are stunningly high-resolution and topologically accurate. But
    mathematically, they add ZERO biological identity over 2D keypoints. Monocular
    models hallucinate *generic, plausible* human geometry; they do not encode
    true identity-specific micro-curvature. The "fast path" is definitively dead.
* **[CONCLUDED — FAIL] Phase 2: Depth Encoder z_d** (`exp/geometry-pca`)
  * Depth (64×64, k=50, FFHQ-fit) adds NO complementary identity signal over z_g.
  * Operational proof: verification AUC z_g=0.541 → +z_d = −0.004 (every mode);
    kNN identity acc 4.3% → −0.2%. Depth slightly *dilutes* the weak geometry signal.
  * **[2026-06-11] FAIL confirmed on face-crop re-run at 24× facial depth
    resolution, domain shift eliminated:** z_g 0.681 → +z_d best delta −0.023
    (−0.034 on seg-clean subset). Resolution & distribution are exhausted as
    excuses; monocular relative depth is conclusively a dead partition.
  * Data defect found: Sapiens seg collapses on ~10% of tight face crops →
    empty seg-masked depth. Controlled via fg≥30% filter (script 36); FAIL
    survives. ⚠️ z_a-on-face-crops must apply the same filter.
  * Metric bug caught + fixed: trace-J = tr(S_B)/tr(S_W) is a weighted average for
    concatenated vectors → blind to complementarity. Gate instrument switched to
    verification AUC. Cross-examined (4 metrics) to rule out a false-fail.
  * Secondary (CORRECTED 2026-06-11): the "z_g=0.54 weak carrier" reading was an
    editorial-keypoint-resolution artifact. Face-crop keypoints, same frozen
    encoder, same images: z_g AUC **0.671** — geometry is a *moderate* carrier;
    confidence ≠ precision (editorial conf was higher, precision lower).

* **[CONCLUDED] Phase 1-R: Pose-Invariant Geometry Encoder** (`exp/geometry-pca`)
  * Shipped 3D-frontalized geometry encoder (z_scale=1.0) on 69,851 FFHQ faces.
  * Real-image hegre Fisher gate: 3D beats flat 2D GPA on aggregate identity
    separability (S_W 314→246, S_B holds). Clean-C1 narrative died at n=10.
  * Geometry alone is a modest identity carrier (J≈0.08) — empirically motivates
    the multi-partition E vector. Contamination near-miss documented as a warning.
* **[SUPERSEDED] Phase 1: Geometry PCA Encoder**
  * 2D GPA left yaw/pitch in C1/C2 (pose-entangled). Superseded by Phase 1-R.
