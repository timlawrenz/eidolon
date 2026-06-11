# Phase 3 — DINOv3 Bridge Implementation Plan

> **For Hermes:** execute task-by-task. Pre-registration (Task 0) MUST land in the
> ledger before running Task 3 or 4.

**Goal:** Validate the premise that DINOv3 semantic embeddings (1024-d cls token)
linearly encode our physically interpretable sliders (z_g geometry, z_a surface).

## Two-Stage Gate (Pre-registered)

### Phase 3: The Premise Test (FFHQ)
Fit `W` via 5-fold cross-validated ridge regression on the ~70k FFHQ cache:
`dinov3_cls -> z_x`.
* **Primary Gate:** Variance-weighted held-out R² **≥ 0.5**, AND per-component
  R² **≥ 0.6 for C1–C10** (coarse geometry must be strongly represented).
* **Falsifiable Prediction:** z_a (micro-surface) will score lower than z_g (coarse
  geometry) because the 16x16-patch DINO token discards fine curvature.
* **Diagnostic Band:** If 0.25 ≤ R² < 0.5, run a 2-layer MLP to diagnose if the
  signal exists nonlinearly.

### Phase 3b: The Transfer Test (hegre)
Apply `W` to hegre `dinov3_cls` to get predicted sliders `Ŷ_a`. Run the canonical
verification-AUC identity test.
* **Gate:** `AUC(Ŷ_a) > 0.5 + 4σ_seed (≈0.51)`.
* Proves the bridge transfers *identity signal*, not just variance, under the
  FFHQ -> hegre domain shift.

## Execution Sequence

1. **Task 1: Hegre Data Completeness (GPU).**
   `stratum process <hegre_raw> --output data/hegre_enriched --passes all`
   Backfills dinov3_cls.npy (and any missing depth/normal/seg/pose) across the
   editorial corpus.
2. **Task 2: FFHQ Bridge Dataset (NAS read).**
   Extract `dinov3_cls` and generate `z_g` (via production encoder) and `z_a_xy`
   (via normal cache + encoder) for the 70k FFHQ set. Save to `data/bridge_dataset.npz`.
3. **Task 3: Fit & Evaluate Phase 3 (FFHQ).**
   RidgeCV, permutation null (shuffled pairs), component-wise R² spectrum.
4. **Task 4: Evaluate Phase 3b (hegre).**
   Gate extractor for dinov3 -> predict Ŷ_a -> verification AUC test.
5. **Task 5: Close-out.**
   Ledger verdict, save `bridge_dinov3_W.npz`, push.