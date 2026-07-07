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
* **[CONCLUDED] Phase 5b: Poser Retrieval Spike — G-A FAIL (informative); GT-LDA ceiling PASS** (`exp/text-to-zg`)
  * **G-A: cross-shoot Prior Recall@k FAIL.** Text→LDA Prior does not beat random-projection
    null at statistical significance (Δ=+0.014, CI[−0.004,+0.033], p=0.063 at k=10,
    n=242 personas). Directionally positive but small — consistent with Phase 5a
    info ceiling. **GT-LDA ceiling PASS:** real held-out-shoot AuraFace → LDA
    hits R@1=0.842 cross-shoot. First proof AuraFace-LDA is a genuine cross-shoot
    identity carrier. Retrieval space is sound; gap lives in the Prior.
  * 🔴 Bug found + fixed: `predict_pin` was not masking T5 padding tokens.
    Masking fix improved Prior R@10 +29% (0.056→0.072) and Δ +7× (0.002→0.014).
  * Hegre coverage: 35,843 T5+AF images, 238 personas, 217 cross-shoot viable.
    Corpus snapshot growing (2,986→2,999 query images between runs).
* **[CONCLUDED] Phase 5a: Text-to-Identity Priors** (`exp/text-to-zg`)
  * G1 (text→z_g): FAIL — corrected ratio ~1.75, worse than predict-mean null.
    z_g is NOT text-predictable; it is pose/expression, supplied at inference, not text.
  * G2 (text→AuraFace-LDA): verif AUC **0.687** (peak @ epoch 35); attribute
    consistency skin **0.889** / hair **0.712**. Text → coarse on-description
    identity region works; ~0.69 is the information ceiling (text is many-to-one).
  * Initial G1=0.015 / G2=0.564 "PASS" verdicts were REVOKED (metric bugs).
  * PREMISE CORRECTION: the 50 z_g values are pose sliders, NOT identity sliders.
    Identity sliders must come from the AuraFace-LDA space (ceiling AUC 0.9998).
  * NOT tested: generalization to unseen *people* (needs multi-image Hegre);
    persona-creator viability (verification is the wrong objective for it).
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

* **[CONCLUDED] Sapiens2 Keypoints Study** (`exp/sapiens2-keypoints-study`)
  * Re-asks the z_g/DWPose identity question with Sapiens2's dense, confidence-scored
    keypoints (vs DWPose 68). Cohort: 25 personas × 15 cross-shoot imgs.
  * **Faithfulness:** Sapiens2 has real per-keypoint confidence (0.90 seen / 0.28 unseen),
    withholds ~22% on occluded/profile faces (DWPose returns all 68 always), 3× more
    stable within-person (0.0076 vs 0.0243). DWPose hallucinates for completeness; Sapiens2 measures.
  * **Identity (cross-shoot verification AUC):** Sapiens2 2D 0.766 vs DWPose 0.688 (+0.077);
    Sapiens2 3D-frontalized 0.734 vs documented z_g 0.67–0.69. Ceiling AuraFace 0.998.
    Instrument validated: DWPose reproduces 0.688 baseline, chance 0.49.
  * **Confound resolved:** template-lift (z_g recipe) on Sapiens2 kp = 0.736 ≈ measured-GPA 0.734
    → gain is 100% keypoint-source, not method. Measured depth adds ~0 identity (echoes dead z_d).
  * **Verdict:** z_g conclusion refined not overturned — landmark shape is a *weak* identity
    carrier (0.73 ≪ 0.998); density buys +0.04. NOT an AuraFace replacement. Candidate role:
    editable-morphology substrate for the Poser (sliders + angle), complementary to AuraFace identity.
  * **Action:** replace DWPose→Sapiens2 pose wherever z_g geometry is computed (strictly better input).
  * **Next:** widen to ~100 personas + persona-level bootstrap CI for a production number.

  * **[CORRECTION 2026-07-07] z_g stays on DWPose.** Fisher-J split (mixed-zg cohort)
    showed Sapiens2 shape has NO identity-blind transient block (0 axes J<0.05; min J≈0.10)
    — its fidelity forecloses identity-blindness, so it CANNOT be a drop-in z_g replacement
    (would break disentanglement). But it's linearly AuraFace-orthogonal (R²=−0.11), so
    Sapiens2 = a NEW complementary shape/morphology stream, NOT a z_g upgrade. Three handles:
    DWPose→z_g (pose), AuraFace-LDA (appearance), Sapiens2 (shape-morphology). Supersedes the
    "replace DWPose→Sapiens2 for z_g" action above.
