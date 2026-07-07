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

## Recommendation
1. **Replace DWPose with Sapiens2 pose wherever z_g-style geometry is computed** — same pipeline, +0.04 identity, 3× more stable, honest per-point uncertainty. A z_g rebuilt on Sapiens2 keypoints is strictly better input.
2. **Do NOT** pursue Sapiens2 geometry as an AuraFace *identity* replacement (0.73 ≪ 0.998; monocular geometry is identity-poor, consistent with FLAME β=0.585).
3. **Do** consider Sapiens2 dense landmarks/pointmap as the **editable-morphology / sculpting substrate** for the Poser product (anatomical sliders + angle control) — a complementary control stream, not the identity carrier. Identity stays with AuraFace-LDA.

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
