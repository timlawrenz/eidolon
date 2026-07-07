# Sapiens2 Keypoints Study

**Branch:** `exp/sapiens2-keypoints-study` · **Date:** 2026-07-07
**Question:** Are Sapiens2 pose keypoints (a) *faithful* (measure, not hallucinate like DWPose) and (b) a stronger identity carrier than the 68-point DWPose landmarks that `z_g` is built from?

## TL;DR

- **Faithfulness: Sapiens2 wins decisively.** Real per-keypoint confidence (0.90 seen vs 0.28 unseen), withholds ~22% of points on occluded/profile faces (DWPose returns all 68 always), and is **3× more stable within-person** (0.0076 vs 0.0243 config scatter).
- **Identity: Sapiens2 keypoints beat DWPose, source-driven and method-independent.** Cross-shoot verification AUC, method held constant:
  - 2D shape: **Sapiens2 0.766 vs DWPose 0.688** (+0.077)
  - 3D pose-removed shape: **Sapiens2 0.734 vs documented z_g 0.67–0.69**
- **But geometry ≪ appearance for identity.** AuraFace-LDA ceiling = 0.998. Landmark geometry is a *weak identity discriminator* regardless of keypoint quality — density buys ~+0.04, not a replacement.
- **Confound resolved:** the 0.734 is NOT an artifact of the frontalization method. Sapiens2 keypoints through z_g's *exact* template-lift recipe give 0.736 (Δ=−0.002 vs measured-GPA). The gain is 100% keypoint-source. Measured 3D depth adds ~0 identity beyond the 2D configuration (echoes the dead z_d path).

## Instrument validation
- DWPose 2D shape reproduces the documented **0.688** ≈ z_g's 0.69 baseline → pipeline trustworthy.
- Label-shuffle chance null = **0.49** → metric floor correct.
- Note: random-projection null is NOT a valid floor for a raw geometric feature (Johnson-Lindenstrauss preserves cosine → projection ≈ feature). Use label-shuffle.

## Cohort
25 personas × 15 cross-shoot images = 375 (4–15 shoots/persona). All with image + DWPose (`pose.npy`) + AuraFace (`.npy`). Manifest: `output/study_manifest.json`.

## Results table

| Feature | Cross-shoot AUC |
|---------|-----------------|
| Label-shuffle chance | 0.49 |
| DWPose 68-kp, 2D shape | 0.688 (=documented z_g) |
| **Sapiens2 kp, 2D shape** | **0.766** |
| z_g / DWPose 3D-frontalized (documented) | 0.67–0.69 |
| **Sapiens2 kp, 3D-frontalized (measured-GPA)** | **0.734** |
| Sapiens2 kp, 3D-frontalized (template-lift, z_g recipe) | 0.736 |
| AuraFace-LDA (ceiling) | 0.998 |

## Recommendation (CORRECTED — see split addendum below)
1. **z_g STAYS on DWPose.** Fisher-J split (mixed-zg cohort) showed Sapiens2 shape has NO identity-blind transient block (0 axes J<0.05) — its fidelity forecloses identity-blindness, so it canNOT be a drop-in z_g replacement without breaking disentanglement. (The earlier "replace DWPose→Sapiens2 for z_g" is superseded.)
2. **Do NOT** pursue Sapiens2 geometry as an AuraFace *identity* replacement (0.73 ≪ 0.998; monocular geometry is identity-poor, consistent with FLAME β=0.585).
3. **Do** adopt Sapiens2 dense landmarks/pointmap as a **NEW complementary shape/morphology stream** for the Poser (anatomical sliders + angle control). It's linearly AuraFace-orthogonal (shape→AuraFace R²=−0.11) — complementary to AuraFace appearance-identity, not redundant, not a z_g upgrade. Guard non-linear leakage with CFG dropout.

## Split addendum (the decisive follow-up)
Sapiens2 keypoints are a *stronger geometry identity carrier* than DWPose (+0.096 AUC) — but that very fidelity means z_g rebuilt on them would NOT be identity-blind. Fisher-J split on a mixed-zg cohort (100 personas): 42 morphology axes (J>0.15), **0 transient axes (J<0.05)**, min J≈0.10; DWPose mean J 0.136 (toward documented 0.06) with only 10 morphology axes. No identity-blind block to carve → z_g cannot move to Sapiens2. But Sapiens2 shape→AuraFace R²=−0.11 (linearly orthogonal). Two opposed roles, one representation can't serve both: DWPose=identity-blind pose control; Sapiens2=complementary shape-morphology stream. Scripts: `fisher_split_sapiens2.py` (low-zg, confounded), `fisher_split_mixed.py` (corrected).

## Caveats / open
- Cohort 25 personas; widen to ~100 + persona-level bootstrap CI for a production number (see experiment tree next step).
- Study code assumes the `/tmp/sapiens2` standalone backbone + local checkpoints (see scripts headers). Not wired into `tools.hegre_dataset` yet — graduation is code+results only.

## Files
- `scripts/extract_study_features.py` — Sapiens2 308-kp + DWPose + AuraFace extraction for the manifest
- `scripts/extract_kp3d.py` — 3D keypoint positions via pointmap lookup
- `scripts/identity_gate_2d.py` — 2D shape verification AUC + confidence calibration (Arm A + G1)
- `scripts/identity_gate_3d.py` — GPA 3D-frontalized gate (G2)
- `scripts/step3_method_control.py` — measured-GPA vs template-lift confound control
- `output/` — feature caches + result JSONs (gitignored)
