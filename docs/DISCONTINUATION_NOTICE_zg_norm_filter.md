# DISCONTINUATION NOTICE — `z_g` norm as a degeneracy filter

**Date:** 2026-09-24
**Status:** DISCONTINUED. Do not re-attempt.
**Arm that killed it:** `experiments/zg_validity/` (`exp/zg-validity`),
ledger entry *z_g Validity Threshold* — verdict **KILL of the belief**.
**Evidence:** `docs/assets/exp/zg-validity/` (`g2_diagnostics.json`,
`review_crosstab.json`, `g1_skeleton_{high,ctrl}.jpg`).

## What is being discontinued

The belief that a **large `z_g` norm indicates a degenerate/failed encoding**
that should be filtered out of the corpus. Concretely, this retires the claim in
`scripts/pipeline/extract_zg_and_averages.py`:

> *"Reject degenerate z_g (DWPose missed eyes/face → wild PCA projection)"* —
> threshold `> 25`

**This is not a claim that the threshold is mis-tuned.** It is a claim that the
threshold is measuring the wrong thing entirely. Do not re-tune it to 15, 20, or
30, and do not re-derive it on a new sample.

## The structural reason it cannot work

The norm is a **pose-atypicality** statistic, not a **validity** statistic. Three
independent measurements establish this, and none of them can be fixed by a
better threshold:

1. **The stated failure mechanism does not occur.** The rationale is *"DWPose
   missed eyes/face"*. Measured over 60 norm > 25 vs 60 controls (norm 8–12):
   **0 missing keypoints and 0 keypoints below 0.3 confidence in either group**,
   mean confidence 0.787 vs 0.948. There is no missed face to detect. The
   detector fires on a phenomenon the rationale does not describe.

2. **The high-norm stratum is not the human reject pile.** All 31,711 corpus
   samples — including 60/60 in the high stratum — are `approved` in the review
   DB. The 215,914 `tainted:extraction_nonface` never entered the corpus. So the
   non-face failure mode is **already handled upstream**, by review, before
   `z_g` is ever computed. The filter is redundant with respect to its own stated
   purpose.

3. **Visual inspection says the alignments are correct.** The reviewer's
   classification of the corrected contact sheets: the high stratum shows
   **accurate landmark alignment** on **atypical poses (often head-inverted)**;
   the control stratum shows strong alignment on normal poses. The strata are
   separated by *pose atypicality*, not by landmark quality. G2 corroborates
   with distance-based, mirror-invariant quantities — `eye_mouth_ratio` 2.695 vs
   1.173 (the control value is textbook-correct for a face), `align_residual`
   1.7×, `iod_norm` 0.209 vs 0.295.

Therefore the high-norm tail is **real pose coverage, correctly encoded**. A
norm filter does not remove garbage; it removes the most extreme — and for a
pose/expression control stream, the most *valuable* — poses. The action the
docstring intends is the opposite of the action the threshold performs.

## What NOT to retry

- **Re-tuning the norm threshold** (`15`, `25`, or any other round number). The
  `<15` figure from the 2026-07-07 Fisher-J work and the `>25` docstring figure
  are both measuring pose atypicality. Reconciling them is meaningless.
- **Adding a confidence gate as a substitute.** Confidence does not separate the
  strata (0.787 vs 0.948, with zero points below 0.3 in either) and the failure
  mode is on correctly-detected points.
- **Using `zg_distance` from the review DB as a validity filter.** Measured:
  approved mean 277.8 vs non-face mean 276.8 — it does not separate faces from
  non-faces.
- **Filtering the high-norm tail "to be safe".** That is the specific error this
  notice exists to prevent. It deletes real pose coverage and silently narrows
  the geometry control's dynamic range.

## Salvage list — what remains valid and useful

- **The `z_g` vectors themselves are sound.** G0: corpus `z_g.npy` bit-identical
  to the `zg/` encoder source over 400 random samples (max ‖diff‖ = 0.0000000000).
  Keep consuming them per-image.
- **The norm is a usable *pose-atypicality* index**, if labelled as such. It
  cleanly separates atypical from normal poses. Do not describe it as validity.
- **`g2_diagnostics.json`** is a ready-made per-sample pose-geometry table
  (`eye_mouth_ratio`, `iod_norm`, `align_residual`, `roll_deg`, confidence stats)
  for the 120 sampled images, and `src/run_gates.py` regenerates it for any
  sample. Useful as a diagnostic, not a filter.
- **The reviewer's mechanism hypothesis is still open (H3):** these poses may sit
  at or beyond the bounds of the Sapiens pose model's training distribution, so
  the *encoding* may be unreliable even though the landmarks are accurate. That
  is a **train/inference** question, not a corpus-cleaning question, and it needs
  its own pre-registered arm. Do not assume this notice settles it.

## Consequence still outstanding (G3)

The retired claim **drives a persona-average filter** in
`extract_zg_and_averages.py` — the threshold was applied to persona averages.
Persona averages were therefore computed over a subsample selected by a criterion
now known to be invalid, and may be biased. **This re-examination has NOT been
done** and is carried forward in the ledger. Retiring the claim without fixing
this leaves the persona averages unjustified.
