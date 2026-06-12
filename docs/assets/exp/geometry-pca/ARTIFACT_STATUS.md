# Artifact Status — `docs/assets/exp/geometry-pca/`

> This file annotates the evidentiary status of persisted JSON artifacts against
> the project's final, post-Phase-4 empirical conclusions (2026-06-11).
> The source of truth is `docs/02_EXPERIMENTS_AND_RESULTS.md` — these
> annotations resolve conflicts between artifact-level verdicts and the final,
> cross-examined record.

---

| Artifact | Artifact Verdict | Final Verdict | Status | Note |
|----------|-----------------|---------------|--------|------|
| `za_gate_results.json` | `PASS`, `best_variant: rot` | **FAIL** (overturned 2026-06-11) | ⚠️ STALE | The PASS was an artifact of the low editorial-keypoint z_g baseline (0.540). When z_g was corrected to face-crop resolution (0.688), normals subtract identity signal (ΔAUC −0.039). See ledger Phase 2b face-crop overturn. |
| `phase3b_transfer_results.json` | `PASS`, `fraction_identity_retained: 1.71` | **UNINFORMATIVE** | ⚠️ STALE | The gate technically passed (AUC > 0.51) but missed the random-projection null: random 50-d DINO projections (0.712 ± 0.007) beat the bridge (0.704). The `fraction_identity_retained: 1.71` is nonsensical. See ledger Phase 3b review control. |
| `phase3_bridge_results.json` | R² values only (no verdict) | **FAIL** on C1–C10 criterion | ✅ ACCURATE | The R² measurements are correct. The original "permutation null" (shuffled Y post-prediction) was vacuous but was later fixed in script 32. See ledger Phase 3 systematic review. |
| `zd_gate_results.json` | `FAIL` | **FAIL** | ✅ ACCURATE | Verdict matches final. Minor: `reaon_method` typo (should be `reason_method`). The trace-J metric used here was later found to be blind to complementarity — but the FAIL verdict was independently confirmed by verification AUC. |
| `zd_verification_auc.json` | `FAIL` | **FAIL** | ✅ ACCURATE | Decisive instrument. Matches the face-crop re-run confirmation. |
| `za_systematic_review.json` | Systematic review (2026-06-10) | — | ℹ️ HISTORICAL | Documents the review that accepted the initial PASS. Contains the seed-noise and variance analyses that were valid for the editorial-keypoint measurement context. The PASS itself was overturned when the measurement substrate changed. |
| `metrics.json` / `metrics_production.json` / `metrics_posenorm.json` | Phase 1/1-R metrics | — | ℹ️ HISTORICAL | Phase 1-R encoder performance snapshots. Still valid for the shipped encoder. |
| `gate_sweep_results.json` | z_scale sweep (10 identities, 136 images) | z_scale=1.0 shipped | ℹ️ VALID BUT THIN | The production z_scale=1.0 decision rests on a 10-identity sweep with no uncertainty estimates. The sweep artifact is corrupt and was deliberately not regenerated. The sweep was never re-run on the full 100-identity corpus. This is acknowledged in the ledger. |
| `za_gate_meta.json` / `za_fit_summary.json` | Metadata and fit summaries | — | ✅ ACCURATE | PCA fit statistics and gate metadata. Factual, not verdicts. |
| `zd_complementarity_diagnostic.json` | Cross-examination of trace-J bug | — | ✅ ACCURATE | Documents the 4-metric cross-examination that independently confirmed the z_d FAIL. |

---

## Missing Artifacts

The following results have **no persisted JSON artifact** under `docs/assets/`:

| Result | Date | Record |
|--------|------|--------|
| Phase 4 masked patch pooling (AUC 0.797) | 2026-06-11 | Only in ledger prose and tables. Expected artifact: `data/phase4_patch_pooling.json` — check if it exists under the NAS `data/` symlink. |
| Phase 2b face-crop overturn (z_a ΔAUC −0.039) | 2026-06-11 | Only in ledger prose and tables. Expected artifact not specified. |
| z_g face-crop correction (0.540 → 0.671) | 2026-06-11 | Only in ledger prose. |

These are the project's most decision-critical results and the least artifacted. If the raw data exists, it should be copied into the assets tree with a descriptive filename.

---

**Last updated:** 2026-06-12 (documentation sync pass)
