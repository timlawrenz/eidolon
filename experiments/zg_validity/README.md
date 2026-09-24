# zg-validity-threshold

**Status: PRE-REGISTERED — NOT YET RUN.** No data has been examined for this arm.
Awaiting explicit approval of the gates below.

---

## The question

`hegre_corpus` ships per-image `z_g.npy` vectors. ~6.7% of them have a norm past
**25** — the value the project's own extraction code calls degenerate. Nobody has
established whether those vectors are *wrong* or merely *extreme*, and the answer
determines opposite actions.

`z_g` is the **geometry stream** — the pose/expression control in the conditioning
stack. It is consumed per-image by the DiT (cross-attention, per-dim basis per the
Arm O fix). If 6.7% of it is garbage, the geometry control is being trained on
garbage. If it is genuine signal, filtering it would delete real pose coverage.

## Why this exists now

`extract_zg_and_averages.py` L130–131 reads:

```python
# Reject degenerate z_g (DWPose missed eyes/face → wild PCA projection)
if np.linalg.norm(z) < 25:
```

That filter is applied **only when computing persona averages** — never to the
per-image vectors the corpus actually ships. Meanwhile the 2026-07-07 Fisher-J work
used a *different* number, capping its cohort at **< 15** "to exclude DWPose-failure
tail". Two thresholds, neither applied to the shipped data, and the ledger contains
no adjudication of either.

## The mechanism (why a high norm is ambiguous)

From `geometry_pca/zg_inference.py:encode_zg`:

```
face_2d --frontalize--> center_and_scale --> align_single --> PCA(50) --> whiten
```

`z_g` is **whitened** PCA coordinates: `(raw - whiten_mu) / whiten_sigma`. Whitening
divides each component by its own standard deviation, which **amplifies the
low-variance, high-index components**. So a large norm means "this sample projects
unusually far along a low-variance axis", and there are two very different reasons
that could happen:

| | Hypothesis | Mechanism | Correct action |
|---|---|---|---|
| **H1** | **Detector failure** | DWPose missed eyes/face or hallucinated points → frontalization/alignment produces a garbage shape → wild projection | **Filter it out** |
| **H2** | **Genuine extremity** | Real extreme pose/expression that frontalization failed to remove → large but *valid* projection | **Keep it** — and instead fix frontalization |

These are not equally likely and the project has evidence pointing both ways. The
docstring asserts H1. But the ledger's 2026-07-07 note measured `z_g → yaw R² = 0.98`,
which means `z_g` **carries pose** — and a high-yaw sample would produce a large
projection through exactly this mechanism. So H2 is live.

**The discriminator is the DWPose keypoints themselves**, not the norm. If the
keypoints behind a high-norm vector are implausible, H1. If they are valid but
extreme, H2.

## Pre-registered gates

Stated before any measurement for this arm. See `provenance.yaml` for the
machine-readable copy.

### G0 — instrument trust (run first; abort if it fails)

Confirm on this arm's own sample that `hegre_corpus` `z_g.npy` is bit-identical to
the `zg/` source tree. Previously established over 600 samples (‖diff‖ = 0.00000000);
re-confirm. **If the corpus is not a faithful copy, STOP** — every downstream
conclusion would be about the copy, not the encoder.

### G1 — mechanism, visual (the core question)

Draw **60 images at norm > 25** and **60 controls at norm 8–12**. For each, render the
DWPose 68-point keypoints over the pixel and classify keypoint plausibility: do the
points sit on the actual eyes/nose/mouth/face outline, or are they missing,
collapsed, or off-face?

Pre-registered decision rule (fixed now, before looking):

| Outcome | Reading | Consequence |
|---|---|---|
| ≥70% of high-norm **implausible** | **H1** — detector failure | A filter is justified |
| ≥70% of high-norm **plausible** | **H2** — genuine extremity | **No filter.** Retire the docstring claim |
| in between | **MIXED** | The criterion must be **keypoint-based, not norm-based** |

Visual verification is mandatory here, not optional — this project's standing rule is
that geometry/identity sets get eyeballed, and the Phase 2b (`z_a`) overturn is the
precedent for a measured result that dissolved under inspection.

### G2 — quantitative corroboration, independent of the visual

Compare high-norm vs control on keypoint-derived diagnostics available in the
pipeline: per-keypoint confidence, count of missing points, inter-ocular distance in
normalized units, and keypoint-bbox scale/aspect. Direction test, pre-registered:

- high-norm shows **worse keypoint quality** → H1
- keypoint quality **equal**, but pose amplitude (yaw/pitch proxy) larger → H2

Report **effect sizes with confidence intervals**, not just significance. A large-n
significance test on a 6.7% subgroup is exactly how a trivial difference gets
promoted to a finding.

### G3 — falsification of the standing claim

Test the docstring's own claim head-on. If H2 holds, then *"norm > 25 = degenerate
(DWPose missed eyes/face)"* is **falsified**, and two things follow: the claim must be
retired from `extract_zg_and_averages.py`, and the persona-average filter it currently
drives is **unjustified** and must be re-examined — persona averages may have been
silently computed over a biased subsample.

### G4 — the decision

Only if a filter is justified, the chosen criterion must be:

1. **computable per-image from data available at inference time** — so the training
   path and the inference path agree. A criterion that needs training-time context
   creates a train/inference mismatch, which is a worse bug than the one it fixes.
2. **applied identically to all splits** — otherwise it becomes a split confound
   (the briefing's §3.4 rule).
3. **stated with a falsifiable justification, not a round number.** If the answer is
   "15" or "25" purely because those numbers already appear in the codebase, that is
   not a criterion.

The `<15` vs `>25` discrepancy must be reconciled explicitly. **At most one of them
can be right**, and the possibility that *neither* is must be on the table.

### G5 — if the corpus changes

Governed as a **data update**, not a new experiment (standing project rule): backup
before the first write and never overwrite an existing backup, idempotent and
resumable, stamped with the basis fingerprint, `_manifest.json` updated, count delta
reported, and **the train/val/test split re-hashed** — because a filter changes split
membership.

## Falsification / KILL condition

If high-norm samples are visually **valid** with valid keypoints **and** their `z_g`
carries genuine pose signal, the arm's premise collapses. The correct action is then
**to change nothing in the data**, and the deliverable becomes the retired docstring
claim plus a `DISCONTINUATION_NOTICE.md` so that nobody re-attempts the filter later.
**A KILL here is a valuable result** — it removes a wrong belief and simplifies the
architecture, which this project has consistently preferred over a marginal gain.

## Differs from

| | |
|---|---|
| `refit-cleaned` | That arm refit the identity basis. This one is about the **geometry** stream and changes no encoder. |
| `ffhq-basis-reproject` | That arm repaired an **encoding** mismatch (wrong basis). This is about **vector validity** on an encoder that is not stale (`encoder_production.npz`, 2026-06-23 — both datasets postdate it). |
| `zg-token-basis-cfg-guard` (prx-tg) | That arm fixed a **config** defect in how `z_g` tokens are fed. This asks whether the **values** are trustworthy. |

## Out of scope

- Changing the `z_g` encoder or the DWPose → Sapiens2 question. That is settled:
  `z_g` **stays on DWPose** (2026-07-07 — Sapiens2's fidelity breaks the
  identity-blindness the disentanglement depends on).
- The `z_g → yaw R² = 0.98` vs *"pose-invariant by construction"* contradiction.
  Real, and it interacts, but it is a **separate question** about what `z_g` encodes;
  this arm asks whether specific vectors are *valid*. If G1/G2 point at frontalization
  failure, that contradiction becomes the natural follow-up arm.
- Any training. This is CPU-only NAS I/O + numpy.

## Expected cost

CPU-only. G0 is a re-run of an existing check; G1 needs a 120-image contact sheet;
G2 is a pass over keypoint files. No GPU, no training, no model.

## Artifacts to be produced

- `docs/assets/exp/zg-validity/` — the G1 contact sheets (high-norm and control)
- `scripts/` — the sampling + keypoint-diagnostic script for this arm
- A verdict in the ledger, and either a filter specification (H1) or a
  `DISCONTINUATION_NOTICE.md` (H2)
