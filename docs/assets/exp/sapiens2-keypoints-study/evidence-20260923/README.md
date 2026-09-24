# Evidence — 2026-09-23 / 2026-09-24 data-integrity investigation

Scripts that produced the numbers cited in
[`docs/briefings/2026-09-23_prx-tg_eidolon-training-brief.md`](../../../../briefings/2026-09-23_prx-tg_eidolon-training-brief.md)
§7 and in the ledger entries for `ffhq-basis-reproject` and the corpus rebuild.

**Why they are committed here:** these were originally written to the agent
scratch directory (`~/.hermes/profiles/eidolon/cache/scratch/`), which is
**pruned after 24h idle**. Governance documents citing numbers whose producing
code no longer exists is exactly the cross-artifact reconciliation failure the
`scientific-experiment-structure` skill forbids. Copied here 2026-09-24 verbatim.

These are **read-only audit scripts** — they measure, they do not write to the
dataset. The one exception is noted below.

## Index

| Script | What it established |
|---|---|
| `which_basis_ffhq.py` | **DECISIVE.** FFHQ `auraface_lda.npy` was bit-exact on the **pre-refit** basis: `‖stored−OLD‖ = 0.00000000` on 6/6 samples, `‖stored−NEW‖ ≈ 154–155`. The premise of `ffhq-basis-reproject`. |
| `check_stream_compat.py` | FFHQ vs hegre identity magnitude mismatch: norm 0.352 vs 1.000 (2.841×) |
| `ffhq_identity_clusters.py` | FFHQ is ~1 image/identity — only 3.2% of 3,000 raw AuraFace vectors have a partner above cosine 0.6 |
| `test_average_discriminative.py` | Persona-average identity index is sound: R@1 = 0.8879 vs chance 0.0031 |
| `diagnose_identity_target.py` | Persona averages are 99.53% collinear pairwise; own-vs-nearest-other margin 0.0029 vs 0.0042 → whitening is load-bearing |
| `compare_identity_semantics.py` | FFHQ vs hegre identity-vector semantics |
| `ffhq_convention.py` | Refit-basis raw coords have norm ~153 (distinguishes conventions) |
| `hegre_lda_norm_audit.py` | 2,220 hegre per-image LDA files (0.75%) still on the old basis |
| `stale_lda_status.py` | All 2,220 stale files are tainted/non-approved (2,219 `extraction_nonface`, 1 `contamination`) → outside every consumed path |
| `zg_audit.py` / `zg_audit2.py` | hegre `z_g` per-dim std 1.916 vs FFHQ 1.138; norm >15 = 20.08% vs 0.49%; >25 = 6.66% vs 0.03%. Corpus `z_g` == `zg/` source bit-exact. **Motivates `exp/zg-validity`.** |
| `zg_whitening_check.py` | Whitening behaviour of the `z_g` encoder |
| `zg_pose_check.py` | `z_g` ↔ pose relationship |
| `completeness_audit.py` | hegre_corpus 100% complete; FFHQ missing 139 `z_g` + 41 `auraface_lda` + 1 `@eaDir` |
| `split_design_data.py` | Persona/set distribution for the proposed train/val/test split |
| `preflight_lda.py` | Pre-flight checks before the LDA refit |
| `verify_basis.py`, `verify_corpus.py` | Post-run corpus/basis verification |
| `missing_zg.py`, `missing_af.py` | Enumerated the missing-stream samples |
| `inspect_extra_dirs.py`, `why_extra_dirs.py` | Traced the corpus orphan dirs |
| `confirm_no_residue.py` | Confirmed the killed first reprojection attempt left **0** stray files |
| `adversarial_ffhq_reproject.py` | **Adversarial pass** for `ffhq-basis-reproject`: full scan of 70,000 dirs, `norm min=max=mean=1.000000000`, 0 degenerate/NaN; random-sample recomputation 0 mismatches |
| `retrofit_provenance_fields.py` | **Writes.** Added `mode` / `agent_model` to all provenance files (2026-09-24 retrofit). Not a dataset audit. |

## Reproducing

All ran from the repo root with `.venv/bin/python`, reading from
`/mnt/nas-ai-models/training-data/`. They require the NAS mounts. None needs a GPU.
