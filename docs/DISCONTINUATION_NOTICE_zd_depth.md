# Project Discontinuation Notice

**Date:** 2026-09-24 (retrospective — arm concluded 2026-06-11)
**Project:** Phase 2 — Depth Encoder `z_d` (monocular depth as an identity partition)
**Status:** DISCONTINUED — KILL
**Ledger:** `docs/02_EXPERIMENTS_AND_RESULTS.md` § [Phase 2] Volumetric Encoder z_d (depth)

## Summary

A 50-d PCA encoder over monocular depth maps (64×64, FFHQ-fit) was tested as a
third identity partition alongside `z_g` (2D keypoints). It added **no**
complementary identity signal over `z_g`, and in operational terms it slightly
*diluted* the weak geometry signal that already existed.

## What We Learned

### Successful components ✅

- The depth extraction pipeline worked and was reusable — depth caches were built
  and later reused by the `z_a` (normals) investigation, so the infrastructure cost
  was paid once.
- The measurement discipline held: the arm used **verification AUC** rather than
  trace-J, which the project had already established is blind to complementarity for
  concatenated vectors. A trace-J reading would have looked positive.

### Failed components ❌

- `z_g` verification AUC 0.541 → +`z_d` **−0.004** (negative in every mode tested).
- kNN identity accuracy 4.3% → **−0.2%**.
- No mode, no k, no FFHQ-vs-hegre fit produced a positive delta.

## Root Cause

**Monocular depth estimation is ill-posed with respect to identity.** A depth model
trained to produce plausible 3D geometry from a single image learns a *generic human
shape prior* — it recovers the structure any human face has, not the micro-curvature
that distinguishes one person from another. The information is not merely hard to
extract; it is **not present in the depth map** in the first place. Adding a
50-d bottleneck over a signal that lacks the target information can only add noise,
which is exactly the small negative delta observed.

## Why We're Sharing This

The tempting error is to read "depth adds nothing" as "our depth encoder was bad."
It was not — the maps were good. A future attempt will be tempted to retry with
higher resolution, a better depth backbone (Sapiens2, Depth Anything, etc.), or a
larger `k`. **All of those will fail for the same reason**, because the ceiling is
set by what monocular depth can contain, not by the encoder. Do not re-attempt
without a *new* hypothesis that explains why depth would suddenly carry
identity-specific information.

Note the same conclusion was independently reached for surface normals
(`DISCONTINUATION_NOTICE_za_normals.md`) — two different monocular geometry
predictions, same root cause.

## Salvage

- Depth caches: retained on NAS, reused by the `z_a` arm before it was also KILLed.
- `provenance_zd_depth.yaml` / `config_zd_depth.yaml`: kept in
  `experiments/geometry_pca/`.
- Scripts `18_fit_zd_encoders.py`, `19_build_depth_cache_singlepass.py`,
  `20_extract_zd_gate.py`, `21_zd_gate.py`, `22_zd_complementarity_diagnostic.py`,
  `23_zd_verification_auc.py`: kept in `experiments/geometry_pca/scripts/`.
- The depth *decoder* path survives in a different role: `z_d` was later reused as a
  **bottleneck** for the DINOv3 bridge (Phase 3) — also KILLed, separately.
