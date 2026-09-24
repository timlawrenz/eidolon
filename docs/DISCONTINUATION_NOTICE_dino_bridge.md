# Project Discontinuation Notice

**Date:** 2026-09-24 (retrospective — arm concluded 2026-06-11)
**Project:** Phase 3 — DINOv3 Bridge (making DINOv3 identity interpretable via geometric sliders)
**Status:** DISCONTINUED — KILL
**Ledger:** `docs/02_EXPERIMENTS_AND_RESULTS.md` § [Phase 3] DINOv3 Bridge (Premise Validation)

## Summary

The idea: DINOv3 CLS/patch features are a strong identity carrier but not
*interpretable*. A bridge was trained to map DINOv3 → the geometric slider space
(`z_g`, `z_a`) so identity could be controlled along human-readable axes. Both the
R² premise (Phase 3) and the identity-transfer test (Phase 3b) failed. The bridge
**degraded its own input**.

## What We Learned

### Successful components ✅

- **The control that killed it produced the project's identity answer.** The review
  control showed raw `dinov3_cls` on face crops = **AUC 0.766**, the strongest
  identity carrier measured on hegre at the time (vs face-crop `z_g` 0.67–0.69,
  `z_d` 0.56). That measurement is what made DINOv3 the identity stream, and it led
  directly to Phase 4's flesh-masked patch tokens (AUC 0.797) — the **settled**
  identity conditioning.
- A reusable null-hypothesis lesson, see Root Cause.

### Failed components ❌

- **Phase 3 (R² premise):** DINOv3 cannot faithfully reconstruct the sliders —
  `z_a` R² = 0.385; `z_g` C6/C11 ≈ 0 (though C6 is plausibly detector noise, J=0.098).
- **Phase 3b (identity transfer):** technically cleared 0.51 but **UNINFORMATIVE**:

  | path | AUC |
  |---|---|
  | raw `dinov3_cls` (face crop) | **0.766** |
  | **random** 50-d DINO projections | **0.712 ± 0.007** |
  | bridge output `Ŷ_g` | **0.704** |

  Any DINO shadow passes. The bridge scored *below* a random projection of its own
  input — it destroys information.

## Root Cause

Two distinct causes, and the second is the more valuable one.

**1. Structural.** DINOv3's identity information does not survive a low-dimensional
*geometric* bottleneck. The bridge's entire purpose was interpretability-via-sliders,
and that 50-d geometric compression is lossy in exactly the identity direction the
project cared about. Interpretability and fidelity were in direct tension, and the
arm tried to buy both.

**2. Methodological — the gate had no null.** The original gate could not distinguish
"learned a faithful mapping" from "preserved some of the input". A random 50-d
projection of the input scored 0.712, above the bridge's 0.704, so the honest reading
was available the whole time and the gate could not surface it. This is the same class
of failure as a validator that scores the wrong thing.

**Follow-on lesson the project extracted:** the correct null depends on the feature.
For a **raw geometric** feature, a random-projection null is *not* valid — the
Johnson–Lindenstrauss lemma preserves cosine, so projection ≈ feature and the null is
artificially high. Use **label-shuffle** there. For a **learned mapping**,
random-projection *is* the right null, because the question is whether training added
anything over the input. Applying the wrong null in either direction gives a confident
wrong answer.

## Why We're Sharing This

- **Do not retry the bridge with a bigger bottleneck or better training.** The
  bottleneck's size is not the problem; the *geometric* nature of the bottleneck is.
  Widen it and you converge on "just use DINOv3", which is what the project did anyway.
- **The durable output is the null rule.** Any gate of the form "a learned mapping
  achieves score X" must report the score of an untrained/random mapping of the same
  input. Without it the gate is not a gate — this arm is the worked example.
- Note the project already has a matching pitfall for the other direction
  (random-projection null on raw geometric features), so both cases are documented.

## Salvage

- **The identity answer:** raw DINOv3 features are the identity carrier → Phase 4
  (flesh-masked patch tokens, AUC 0.797, cross-shoot verified) → the settled
  conditioning stack. This is the arm's real product.
- Scripts `29_build_bridge_dataset.py`, `30_fit_dinov3_bridge.py`,
  `31_evaluate_dinov3_transfer.py`, `32_phase3_systematic_review.py`,
  `33_extract_face_dino.py`, `34_evaluate_dinov3_face_transfer.py`,
  `35_hegre_fit_bridge_control.py`, `37_dino_patch_face_pooling.py`: kept in
  `experiments/geometry_pca/scripts/`.
- `provenance_dino_bridge.yaml` / `config_dino_bridge.yaml`: kept in
  `experiments/geometry_pca/`.
- Results JSON: `docs/assets/exp/geometry-pca/phase3_bridge_results.json`,
  `phase3b_transfer_results.json`.
