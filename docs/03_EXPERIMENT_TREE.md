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
* **[CONCLUDED — KILL (of the belief)] z_g Validity Threshold (`exp/zg-validity`)** 🪦
  * **Question:** ~6.7% of `hegre_corpus` per-image `z_g` has norm > 25 — the value the
    project's own extraction code calls degenerate. Are those vectors *wrong* (DWPose
    failure) or merely *extreme* (genuine pose)? The two readings imply **opposite
    actions**, and the geometry stream is consumed per-image by the DiT.
  * **Answer: they are merely extreme. The claim *"norm > 25 = degenerate"* is FALSIFIED.**
    `z_g` norm is a **pose-atypicality** index, not a validity index. Retired claim +
    structural reason: `docs/DISCONTINUATION_NOTICE_zg_norm_filter.md`.
  * **Evidence (run 2026-09-24, CPU-only):**
    * G0 **PASS** — corpus `z_g` bit-identical to the `zg/` encoder source
      (400 samples, 0 mismatches, max ‖diff‖ = 0.0000000000).
    * **The stated mechanism does not occur:** 60 norm>25 vs 60 controls (8–12) →
      **0 missing keypoints, 0 keypoints below 0.3 confidence in either group**,
      mean confidence 0.787 vs 0.948. Nothing was "missed".
    * **The stratum is not the human reject pile:** all 31,711 corpus samples — and
      60/60 in the high stratum — are `approved` in the review DB. The 215,914
      `tainted:extraction_nonface` never entered the corpus, so the non-face failure
      mode is already handled upstream. The filter is redundant with its own purpose.
    * **Visual (corrected sheets):** high stratum = accurate landmark alignment on
      **atypical poses (often head-inverted)**; control = strong alignment, normal
      poses. Separated by pose atypicality, not landmark quality → **H2 → no filter.**
    * **G2 corroboration (distance-based, mirror-invariant):** `eye_mouth_ratio`
      **2.695 vs 1.173** (control is textbook-correct for a face), `align_residual`
      0.434 vs 0.253, `iod_norm` 0.209 vs 0.295.
  * **Instrument defect found and fixed mid-run:** the first G1 render applied a
    **vertical mirror** to the pose points; that render and its visual read were
    discarded. G2 was unaffected (it reads raw pose coords, never the pixel mapping).
    Fix: `src/run_gates.py::_denormalize`, commit `c0ed696`. Caught by the user —
    my own "points are centred?" check was blind to it (a mirror preserves the
    centroid). Orientation is now verified by anatomical ordering.
  * **Carried forward:** G3 (re-examine the persona-average filter the retired claim
    drives — persona averages may be biased); H3 (is the *encoding* reliable at
    pose-distribution extremes? the reviewer's Sapiens-OOD hypothesis — needs its own
    gate); a corpus-quality arm (extreme-pose/off-frame crops pass review and affect
    **all three streams**); and a `tainted:approved_bad_geometry` constant-distance
    artifact (19 rows, bit-identical `zg_distance`).
  * Arm dir: `experiments/zg_validity/` — evidence in `docs/assets/exp/zg-validity/`
* **[CONCLUDED — PASS] FFHQ Basis Reprojection (`exp/sapiens2-keypoints-study`)**
  * **Problem:** `ffhq/stratum/{id}/auraface_lda.npy` is on the **PRE-refit** LDA basis
    (files 2026-06-30; basis refit 2026-07-23). Proven bit-exact: `‖stored − project_old(raw)‖ = 0`
    on every sampled file, `‖stored − project_new(raw)‖ ≈ 153`.
  * **Consequence:** `prx-tg/production/data_stratum.py` loads `auraface_lda.npy`
    directly with no basis check, so every `eidolon`-adapter arm whose `stratum_dirs`
    included FFHQ (weight 2.3) fed a 64-d identity slot **two incompatible encodings**
    (FFHQ pre-refit norm 0.35 / hegre-corpus refit norm 1.0). The identity conditioning
    of all five `exp/eidolon-conditioning` arms is therefore confounded — including
    Arm O's PASS, whose *geometry* result stands independently.
  * **Fix:** reproject 69,960 files onto the refit basis + L2-normalize (matches
    `hegre_corpus`), with a pre-refit backup and a `BASIS_FINGERPRINT.json` guard so
    loaders can refuse mixed-basis input.
  * **Result (2026-09-24): 69,960 written, 0 errors, 536 s, CPU-only.** Adversarial pass
    on a FULL scan of 70,000 dirs: 69,960 at unit norm, 0 degenerate, 0 NaN,
    `norm min=max=mean=1.000000000`; random-sample recomputation 0 mismatches.
    Backup 69,960 entries (36.5 MB). Stamps written for `ffhq/stratum`,
    `hegre_corpus`, `hegre-faces/v1/lda`. Basis fingerprint `120e1c5a1dc4f423`.
  * **Not fixed by this arm:** FFHQ is still ~1 image per identity, so its identity
    vectors still teach "identity vector = per-image key" — encoding-consistent,
    not a valid identity target.
  * Arm dir: `experiments/ffhq_basis_reproject/`
* **[CONCLUDED] Phase 5b: Poser Retrieval Spike — G-A FAIL (informative); GT-LDA ceiling PASS** (`exp/text-to-zg`)
  * G-A: cross-shoot Prior Recall@k FAIL. Text→LDA Prior does not beat random-projection
    null at statistical significance (Δ=+0.014, CI[−0.004,+0.033], p=0.063 at k=10,
    n=242 personas). Directionally positive but small — consistent with Phase 5a
    info ceiling. **GT-LDA ceiling PASS:** real held-out-shoot AuraFace → LDA
    hits R@1=0.842 cross-shoot. First proof AuraFace-LDA is a genuine cross-shoot
    identity carrier. Retrieval space is sound; gap lives in the Prior.
  * 🔴 Bug found + fixed: `predict_pin` was not masking T5 padding tokens.
    Masking fix improved Prior R@10 +29% (0.056→0.072) and Δ +7× (0.002→0.014).
  * Hegre coverage: 35,843 T5+AF images, 238 personas, 217 cross-shoot viable.
    Corpus snapshot growing (2,986→2,999 query images between runs).
  * **[2026-07-20] GT-LDA ceiling raised to R@1=0.854** after LDA basis refit
    on cleaned 324-persona dataset (2.1× personas, 3× images).
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
  * 🪦 **Tombstone: [`docs/DISCONTINUATION_NOTICE_dino_bridge.md`](DISCONTINUATION_NOTICE_dino_bridge.md)** —
    root cause: DINO identity does not survive a low-dimensional *geometric*
    bottleneck (the bridge scored 0.704, **below a random 50-d projection at
    0.712**). Origin of the mandatory-null rule: a gate of the form "a learned
    mapping achieves X" must report the untrained/random mapping's score.
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
  * 🪦 **Tombstone: [`docs/DISCONTINUATION_NOTICE_za_normals.md`](DISCONTINUATION_NOTICE_za_normals.md)** —
    this arm is the project's canonical *false PASS* (seg-collapse contamination).
    Origin of the "always verify identity test sets visually" rule.
* **[CONCLUDED — FAIL] Phase 2: Depth Encoder z_d** (`exp/geometry-pca`)
  * Depth (64×64, k=50, FFHQ-fit) adds NO complementary identity signal over z_g.
  * Operational proof: verification AUC z_g=0.541 → +z_d = −0.004 (every mode);
    kNN identity acc 4.3% → −0.2%. Depth slightly *dilutes* the weak geometry signal.
  * 🪦 **Tombstone: [`docs/DISCONTINUATION_NOTICE_zd_depth.md`](DISCONTINUATION_NOTICE_zd_depth.md)** —
    root cause: monocular depth learns a *generic* human-shape prior, so identity
    information is not present to extract. Do not retry with a better depth model.
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

---

## `zg-identity-blindness` — `[CONCLUDED — PARTIAL]` (2026-09-24)

**Branch:** `exp/zg-identity-blindness` · **Mode:** `confirmatory` · CPU only, no GPU.

**Question:** does `z_g` carry almost no identity, once measured on the *curated*
corpus with a *rebuilt* instrument?

**Result.** G0 PASS (corpus `z_g` bit-identical to source). **G1 FAIL** — the corpus
`auraface_lda` is the **persona centroid** (321/321 personas bit-identical across all
their samples; `S_W = 0` by construction, so J is undefined on it). **G1b PASS** on
the correct array (per-image LDA): **J = 2.0137 = 197.5× the noise floor vs `z_g`'s
8.31× — a 24× separation**, so the instrument does detect identity separability.
**G2 PARTIAL**: `z_g` J = **0.0847** (venue B), morph axes **11** — inside the J band
but **one axis over** the pre-registered ≤ 10. **G3 `UNREPRODUCIBLE`** (original
producing script absent from every branch; no stand-in substituted).

**Size-corrected, `z_g`'s relative identity signal FELL** (12.60× → 8.31× floor).
Raw J rose only because the corpus shrank: the floor `(C−1)/(N−C)` is **2.18×
higher** at 31,711 samples than at 69,110, so raw J rises for free. **The J = 0.059
in the prx-tg briefing is superseded** — no producing script, and size-confounded.

**Verdict: PARTIAL** — the belief ("identity lives in AuraFace; `z_g` is a
geometry/pose control space") **stands in direction**, magnitude corrected. Not
smoothed into CONFIRM.

**Evidence:** `docs/assets/exp/zg-identity-blindness/fisher_metrics.json`
**Code:** `experiments/zg_identity_blindness/src/run_fisher.py`
