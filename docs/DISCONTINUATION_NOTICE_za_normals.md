# Project Discontinuation Notice

**Date:** 2026-09-24 (retrospective — arm concluded 2026-06-11)
**Project:** Phase 2b — Surface Normals Partition `z_a`
**Status:** DISCONTINUED — KILL
**Ledger:** `docs/02_EXPERIMENTS_AND_RESULTS.md` § [Phase 2b] Surface Normals Partition z_a

## Summary

A 50-d encoder over monocular **surface normals** was tested as a complementary
identity partition. It appeared to PASS initially, then was **overturned** by a
face-crop re-run: `ΔAUC −0.039` against baseline `z_g`. The overturn is itself the
most valuable output of this arm.

## What We Learned

### Successful components ✅

- **The overturn discipline worked.** This arm is the project's canonical example of a
  measured PASS that was actually a measurement artifact. Catching it is why the
  adversarial pass is mandatory before any PASS verdict in this project.
- The contamination mechanism was identified precisely: the original test used
  face-crops that suffered **seg-collapse**, and the re-run applied an `fg ≥ 30%`
  filter to prevent it. The project's standing rule "always verify identity test sets
  visually" traces directly to this arm.
- A genuinely useful negative: normals are **high-resolution and topologically
  accurate** on face crops — visually stunning — which is exactly what made the false
  PASS convincing.

### Failed components ❌

- Baseline `z_g` (0.688) → +`z_a` (0.649) = **ΔAUC −0.039 (FAIL)** on the seg-clean
  subset.
- The apparent complementarity disappeared entirely once contamination was removed.

## Root Cause

**Same as `z_d`, and this arm proves it more sharply: monocular geometry prediction
hallucinates *generic, plausible* human structure rather than encoding true
identity-specific micro-curvature.** The visual quality of the normals is a red
herring — a high-resolution, topologically accurate normal map can still contain zero
biological identity information, because the model's inductive bias is "a plausible
human face", not "this person's face". The mathematical contribution over 2D
keypoints is zero.

**Secondary methodological root cause:** the arm's gate was measurable but its *test
set* was contaminated (seg-collapse), and nothing in the process at the time forced a
visual check. A correct metric on a corrupt test set is still a wrong answer.

## Why We're Sharing This

Two lessons, both generalizable:

1. **Do not re-attempt normals with a better model.** Sapiens2 normals, or any other
   monocular normal predictor, will fail identically. The ceiling is what monocular
   normals can represent, not which network produces them. Note Sapiens2's shape
   stream *was* later found valuable — but as a **dense-keypoint morphology stream**,
   not as normals, and not as an identity partition.
2. **A "stunningly good-looking" intermediate is not evidence.** If you find yourself
   arguing from how good the normals look, you are about to repeat this arm's false
   PASS. Argue from the verification AUC on a visually-verified, seg-clean test set.

## Salvage

- The `fg ≥ 30%` seg-clean filter — now a standing requirement for any face-crop
  experiment in this project.
- Scripts `24_build_normal_cache_singlepass.py`, `25_fit_za_encoders.py`,
  `26_extract_za_gate.py`, `27_za_gate.py`, `28_za_systematic_review.py`,
  `36_zd_facecrop_seg_control.py`: kept in `experiments/geometry_pca/scripts/`.
- `provenance_za_normals.yaml` / `config_za_normals.yaml`: kept in
  `experiments/geometry_pca/`.
- Normal caches: retained on NAS.
