# Briefing — Eidolon → prx-tg: the persona-vector visualization experiment

**Date:** 2026-09-24
**Subject:** The experiment that renders a persona's identity vector into pixels.
**Status of this document:** current state + standing consequences. This is not a
repair log — the history lives in the ledger, tree and `PROJECT_STATUS`.

---

## 0. Purpose

Eidolon's job is to **identify a vector that expresses a person's identity**. That
vector is settled: **AuraFace, LDA-compressed to 64-d**. prx-tg's job is to be the
**visualization tool** that renders that vector — take this person's identity, show
them in another pose, and have the identity in the returned image still be *that
person*.

This briefing specifies that experiment against the current state of both projects:
what to condition on, what the data contract is, what is already proven, what is
blocked and on what, and the gates.

Companion document — **read both**:
`2026-09-23_prx-tg_eidolon-training-brief.md` (the data/gates handoff). This
briefing does not repeat its data-status audit or its G1–G5 gate definitions; it
specifies the *arm* those gates apply to.

---

## 1. The conditioning contract

The request, verbatim in intent: **train on AuraFace + `pose.npy`; if DINO is
included, DINO CLS tokens — not patch tokens.** The goal is that changing pose
changes the pose and *not* the identity, and changing identity changes the identity
and *not* the pose.

| stream | source | role | status |
|---|---|---|---|
| **identity** | `auraface_lda.npy`, 64-d | the persona vector — what we are visualising | settled |
| **pose/expression** | `pose.npy` (DWPose) → `z_g`, 50-d | the variable we sweep | settled |
| **DINO** (optional) | **CLS token** (1024-d) | detail/style carrier | conditional — see 1.2 |
| text (T5) | — | **not in the render path** | out by design |

### 1.1 Why `pose.npy`/DWPose and not Sapiens2 `pose2`

This is a settled decision with a measured basis, and it should not be re-opened
casually. Sapiens2 dense keypoints are the **better identity carrier** — cross-shoot
AUC 0.766 vs DWPose 0.688, a +0.096 gap that widened at 4× cohort. That is precisely
why they are **wrong** here:

> Fisher-J axis split on a mixed-z_g cohort: Sapiens2 mean **J = 0.331** (42
> morphology axes) vs DWPose **0.136** (→ 0.059 on the full corpus). **No
> identity-blind transient block exists in Sapiens2 shape** — every axis is
> person-discriminative. Dense keypoints capture identity so well that *how a face
> moves is itself identity.* You cannot carve an identity-blind `z_g` out of
> Sapiens2 by axis selection.

Identity leakage in the pose stream *breaks the orthogonality the product depends
on*. `pose.npy` carries less identity; that is the feature. prx-tg has a
`stratum2-pose2` config — it exists, and it is not the path for this arm.

**Corollary for the record:** the three models are distinct and must not be
conflated. **DWPose** writes `pose.npy`. **Sapiens 1B** writes `depth/normal/seg`
(the KILLed `z_d`/`z_a` inputs). **Sapiens2** writes `pose2.npy` in the separate
`stratum2/` tree. An out-of-distribution question about `z_g` is a question about
**DWPose's** training distribution (COCO-WholeBody — upright, largely frontal
subjects), not Sapiens'.

### 1.2 DINO CLS — one standing warning

"Sapiens2 keypoints carry identity" and "DINO patch tokens carry identity" are the
same problem in different clothes. prx-tg's own prior findings:

> **DINO patches buy reconstruction fidelity (+0.043 LPIPS) at the cost of CLIP
> score, aesthetics, face confidence and text-following.** DINO CLS alone is
> sufficient style conditioning.

So CLS-over-patches is the right call for this arm. But CLS is **the memorization
vector** — the diagnosed taint was the model "memorizing CLS tokens as per-image
keys into the training set", with CLS present on ~45% of steps and text+CLS
co-occurring only ~40%. **Consequence: any DINO CLS conditioning in this arm must
be paired with an explicit memorization probe** (novel-CLS / held-out-CLS at eval).
Please do not add CLS "to improve quality" without that probe — that is exactly the
shape of the failure that got P1/PP cancelled.

---

## 2. What is already proven, and what it obliges

**The pose conditioning works, and the failure mode that stopped it is solved.**
This is not history to re-derive; it is a constraint on the next arm.

The first attempt at this exact architecture — AuraFace-LDA identity via adaLN +
`z_g` geometry via cross-attention, T5 and DINO dropped — rendered **all faces
frontal** even though `z_g` encodes head yaw at R² = 0.996. Data augmentation did
not fix it. The cause: the 50 `z_g` tokens were built by a **single shared
`Linear(1,H)` MLP with no per-dimension identity**, making them permutation-symmetric
so cross-attention could not address a specific `z_g` axis. *The geometry stream was
nearly inert.*

**Standing consequence — the recipe is load-bearing, not a tuning detail:**

1. **Per-dimension `geo_basis` tokens** (so `dim0 → yaw` is addressable).
2. **CFG-guarded basis activation** — the basis must not fire on CFG-dropped steps.
   Applying it unconditionally is a pure noise injection (~30% of steps) and
   produced mode collapse at step 3500 with no loss or gradient spike.
3. **Asymmetric CFG dropout on the identity stream.** AuraFace leaks pose
   (yaw R² ≈ 0.46, pitch R² ≈ 0.64). Without identity-stream dropout the DiT cheats
   pose out of the identity vector and `z_g` binding dies.

This combination is **Arm O — `[CONCLUDED — PASS]`**: monotonic `dim0 → yaw` at
steps 3000 *and* 5000, zero collapse, loss 0.0099, recon LPIPS 0.819. First arm ever
to hold controlled yaw without collapse. Release artifact
`release/prx_tg_armo_cfgguard_step5000_bf16.safetensors`.

**Arm O proves the architecture can bind geometry. It does not prove identity
retention or disentanglement** — those gates are still unmeasured at the renderer.
That gap is the substance of this arm.

---

## 3. The blocked premise — current state

**The VAE high-fidelity hypothesis is CONFIRMED.** P1/PP were a deliberately quick
spike on dirty data, and what they set out to verify — the fidelity reachable by
going through a VAE — is now established. **Do not read the kill as a dead
hypothesis.**

**What was killed is the arm, not the hypothesis**: the assets that spike produced
do not meet the data-quality and provenance gates, so neither checkpoint may
initialize a future arm. The fidelity finding stands; those specific weights are
unusable.

The v1 visualization plan (`2026-09-21_2320-eidolon-visualization-engine.md`) assumed
Arm PP as the backbone donor. **That donor premise is dead** — there is no
warm-startable checkpoint — and the plan is marked superseded, do not execute.

**Consequence:** this arm now sits downstream of prx-tg's gated redesign, as
**Stage E** in that sequence (Stage A forensic baseline → B data prep + split →
C gated P1 → D gated PP → **E eidolon renderer**). It cannot warm-start from a
tainted lineage, and it should not be launched ahead of Stage C/D producing a clean
gated backbone — otherwise the identity-retention measurement has no trustworthy
substrate and the arm repeats the P1/PP error in a new place.

Arm O remains the code and recipe reference; it is not a weight donor.

---

## 4. The data contract — decided

This was the sharpest open question and it is now settled (§8, decisions 2 and 3).
The reasoning below is why the answer is what it is.

**Identity persistence — the product claim — can only be trained and measured with
multiple images per identity.** FFHQ is 1 image/identity; it cannot express "same
person, different pose". Hegre is the **only** cross-shoot substrate in the project:
31,711 samples, 321 personas, multiple shoots per persona.

**Why hegre persona centroids must be in the training set — and why FFHQ alone
cannot work.** FFHQ has **no identity overlap**: every identity vector appears
exactly once. Trained on FFHQ alone, the model has no pressure to learn an identity
*representation* at all — the cheapest solution that fits is a **1:1 AuraFace →
image lookup table**, one entry per training sample. It will reconstruct its
training identities beautifully and generalise to no one, and no FFHQ metric can
detect the difference, because FFHQ cannot ask "render *this person* again in a
different pose".

Adding **hegre persona centroids breaks the lookup table by construction**: one
identity vector now appears across many poses. The only way to fit those pairs is to
learn a *representation* of that identity that survives a pose change — which is
exactly the product requirement expressed as a training constraint. **This is not a
data-enrichment convenience; it is the mechanism that forces the model to learn
identity rather than memorise a mapping.**

A holdout remains fully feasible and useful on top of this: hold out whole
**personas**, so identity persistence is measured on people the model has never
seen.

**But hegre is also the gate instrument.** Training on it directly destroys the only
held-out identity-persistence measurement that exists. That is a genuine tension,
not a process formality.

**Resolution already chosen by prx-tg, to be honoured here:** *"hegre split to be
vertical (person-level holdout) when it binds at Stage E."* Train on a subset of
**personas**, hold out whole personas. Then identity persistence is measurable on
people the model has never seen — the strongest form of the test.

### 4.1 The centroid construction (the "enriched dataset" idea)

The approach that was raised and should be the basis of the design: **condition on a
persona's AuraFace centroid, with many pose combinations varying around it.**

- Identity conditioning value = the **per-persona AuraFace centroid**
  (`hegre-faces/v1/averages/*.auraface.npy`), not the per-image vector.
- Pose conditioning varies across that persona's real samples.
- Training then directly supervises "same identity vector + different pose →
  images of that same person", which is the product requirement stated as a loss.

Two consequences worth stating before implementation:

1. **Do not mix identity conventions.** FFHQ's `auraface_lda` is per-image
   (70k identities); hegre's is person-averageable. Mixing per-image and
   centroid identity vectors in one run is a confound. If FFHQ is used at all, keep
   the conventions distinct and say so in `diff_summary`.
2. **Known data hazard at this exact junction.** All-zero rows / zero-keypoint
   samples poison centroids — the DWPose zero-keypoint pitfall. Audit the centroid
   inputs for all-zero and degenerate rows *before* they enter training, not after.
   (The orphan persona `hera` — a `z_g` centroid with no AuraFace centroid — is the
   same family of defect; it has since been repaired, but the class of bug persists.)

---

## 5. Eidolon-side readiness — what is settled as of 2026-09-24

Stated so prx-tg can rely on it:

- **Identity data is on one basis, one convention.** All identity streams
  (hegre_corpus, ffhq/stratum, hegre-faces/v1/lda) sit on the pooled refit basis,
  L2-normalised, stamped with fingerprint **`120e1c5a1dc4f423`**. FFHQ's stored
  per-image `auraface_lda.npy` were on the **pre-refit** basis and have been
  recomputed and verified (69,960 files, 0 errors, unit norm, adversarial scan
  clean). The mixed-basis defect that silently confounded an earlier generation of
  arms is closed in the data.
- **The corpus is clean and fully reviewed.** All 31,711 corpus samples are
  `approved` in the review DB; the 215,914 `tainted:extraction_nonface` never
  entered it.
- **`z_g` is cleared for per-image use — with no filter.** The high-norm tail
  (6.66% at norm > 25) is **not** a degeneracy artifact. Measured on 60 norm>25 vs
  60 controls: **0 missing keypoints and 0 sub-0.3-confidence keypoints in either
  group**; the strata are separated by **pose atypicality** (atypical, often
  head-inverted poses with *accurate* landmark alignment), not by landmark quality.
  The `norm > 25 = degenerate` claim is retired
  (`docs/DISCONTINUATION_NOTICE_zg_norm_filter.md`). **Filtering this tail would
  delete the most extreme real pose coverage** — the opposite of what a pose sweep
  needs. Consume `z_g` per-image, unfiltered.
- **Geometry binding is instrumentally verified:** `z_g → yaw R² = 0.996` on the
  full corpus, cleanly isolated on `dim0`.
- **Orthogonality, representation level:** ridge `z_g → AuraFace` R² = **−0.033**
  (8k FFHQ pairs) — the streams are genuinely complementary; "project `z_g` out of
  AuraFace" was dropped as having nothing linear to remove. `z_g` carries almost no
  identity (Fisher J = **0.059**).
  ⚠️ **Caveat that matters to your gates:** that J was measured while the corpus was
  still curating (unreviewed/`bad_geometry` noise inflates within-person scatter,
  the denominator of J) and is flagged in the ledger as *directional, not final*.
  Re-measurement on the now-clean corpus is pending on the Eidolon side. Treat the
  number as provisional in gate thresholds.

---

## 6. Gates

**G0–G5 are defined in `2026-09-23_prx-tg_eidolon-training-brief.md` §4** and are not
repeated here. In outline: G0 photorealism · **G1 identity binding (the missing
instrument)** · G2 geometry binding (Arm O's gate, inherited) · **G3
disentanglement / non-interference (the core Eidolon claim)** · G4 memorization
ceiling · G5 stability.

Two additions specific to the visualization arm:

- **G6 — cross-shoot identity persistence (the product question).** Render from
  held-out **persona centroids**, re-embed with AuraFace, measure **cross-shoot
  retrieval R@1** against that persona's *other shots*, with a persona-level
  bootstrap CI. No FFHQ metric can answer "can I render *this person* from their
  vector" — only this can. This is the measurement R1/hegre-holdout exists to
  protect.
- **Arm O's gate inherits verbatim:** monotonic DWPose-measured yaw on the `z_g`
  `dim0` sweep at ≥2 checkpoints, no collapse.

And the binding constraint on anything involving CLS: **a novel-CLS / held-out-CLS
probe must ship with the arm** (§1.2).

---

## 7. What would make this arm uninformative

Written before results, so it cannot be rationalised later:

- Launching before Stage C/D yield a clean gated backbone → identity retention is
  measured on an untrusted substrate and the result is uninterpretable.
- Training on FFHQ alone → identity persistence is *unrepresentable* (1 image per
  identity). Any "identity holds" result is an artifact of the training
  distribution, not a property of the renderer.
- Measuring identity retention against *training* identities → memorization is
  indistinguishable from retention. Held-out personas only.
- No CLS probe while CLS is in the conditioning → cannot separate conditioning from
  key-lookup memorization.
- Reading a `z_g` sweep that "changes pose but not identity" without checking the
  pose actually changed (DWPose-measured yaw) → an inert stream also produces no
  identity drift, and looks like success.

---

## 8. Decisions (agreed 2026-09-24)

All six open questions are resolved. Recorded here so Stage E is specified rather
than negotiated:

1. **Sequencing — requirements now, numbers at Stage E.** Pre-register the
   *requirements* now (persona-level holdout; held-out personas reserved and never
   touched) so Stage C/D shape the split to feed Stage E. Freeze the numeric
   thresholds at Stage E.
2. **Data contract — hegre persona centroids go IN the training set.** Not optional
   and not merely enrichment: FFHQ has no identity overlap, so FFHQ alone admits a
   1:1 AuraFace → image lookup table as the cheapest fit. Centroids break that
   construction (§4). **Person-level holdout confirmed** — and on top of the
   centroids it is both *feasible and useful*.
3. **Identity conditioning value = the persona AuraFace centroid** — for now. The
   per-image drift check (does the render match this persona's *individual* shots,
   not just their mean?) is deferred; note that it is the measurement that would
   distinguish "renders this person" from "renders the average of this person".
4. **DINO — out of this arm.** No patch tokens, and no CLS either. The core arm
   stays a two-stream test of identity-vs-pose; CLS is a single-variable follow-up
   with its mandatory novel-CLS probe, not part of this run. (The CLS warning in
   §1.2 therefore does not gate this arm — it gates any future arm that adds it.)
5. **Warm start — fresh P1.** Stage E starts from a **fresh P1**; the VAE route is
   good enough for now. Explicitly *not* the killed P1/PP assets — those weights
   fail the data-quality and provenance gates and may not warm-start anything.
6. **Budget — 10k steps at 1k-segment granularity.**

---

## 9. Bottom line

Eidolon's identity vector is settled and its data is clean, unified and cleared for
per-image use. The pose conditioning path is **proven to work** (Arm O) and its one
historical failure — inert geometry producing all-frontal faces — is root-caused and
fixed by a recipe that must be carried forward intact, not re-derived. The VAE
fidelity question is **answered**; only the spike's weights were unusable.

The experiment is now specified end to end: two streams, no DINO, persona centroids
in the training set against a persona-level holdout, warm-started from a fresh P1,
10k steps. The one open dependency is prx-tg's own sequence — Stage C/D must yield
the fresh P1 and the split this arm's measurements require.

What remains is not an idea and not a contract; it is a **clean backbone to
warm-start from**. This briefing pins down the conditions under which the
visualization experiment is informative — so that when Stage E runs, it answers
*"does this vector express a person?"* rather than re-discovering why it did not.