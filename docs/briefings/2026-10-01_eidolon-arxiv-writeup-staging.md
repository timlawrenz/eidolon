# Eidolon — scientist-facing technical write-up: STAGING BRIEF

**Date:** 2026-10-01
**Status:** `[STAGED — draft not written]` · **staging only; nothing public**
**Owner:** eidolon-publication driver (cron `6f62e66fb97b`, currently PAUSED)
**Source of truth:** `docs/02_EXPERIMENTS_AND_RESULTS.md` (the ledger). Every number
below is quoted from the ledger, not recalled.

> This is the **outline / staging artefact** for eidolon's first scientist-facing
> release. It exists to catch framing drift cheaply, before a full draft is written.
> It is not itself the artefact, and it releases nothing.

---

## 0. Defaults taken (disclosed — the review questions timed out, silence executes the default)

| Decision | Default taken | Why |
|---|---|---|
| **Form / destination** | Staged **arXiv-style technical report in Markdown + figures under `docs/briefings/`**; actual arXiv submission decided only after founder review | Matches the registry `artefact_dir: docs/briefings/` and the driver's staging-only rule; reversible |
| **What it leads with** | **Method + ablations** ("a disentangled face conditioning stack"), with the negative results as a credibility section | The ledger's positive core is the two-vector conditioning stack; the negatives are strongest as support |
| **Relation to the existing 2026-07-01 blog post** | **New standalone write-up**; leave the published post in place and link forward | Never edit published science without explicit sign-off |
| **Review flow** | **Outline first (this document) → sign-off → full draft** | Cheapest point to catch framing drift |

No file leaves the repo; the driver stays paused; the one-number stays `0`.

---

## 1. The gate this artefact must pass

From `04 - projects/eidolon/Open Questions — audience.md` and `State.md`:

- **audience:** researchers in the field (face identity, disentangled generation, graphics).
- **gate test:** *could a researcher evaluate the claim in 10 minutes — a reproducible
  result with eval numbers, not a marketing post?*
- **external witness (the metric):** the live URL answers 200 / the HF API — never self-reported.
- **presentation matters** explicitly ("look cool"): legible figures, house-style diagrams.

---

## 2. Working title & lead claim

**Working title:** *A Disentangled Face Conditioning Stack: a Pose-Invariant Geometry
Control and a Compact Identity Carrier — and the Negative Results That Shaped Them.*

**Lead claim (one sentence):** facial *pose/expression* and facial *identity* can be
separated into two empirically-orthogonal vectors — a 50-d pose-invariant keypoint
geometry control (`z_g`) and a 64-d identity carrier (AuraFace-LDA, cross-shoot
R@1 = 0.8538) — and the popular "fast paths" to structured identity geometry
(monocular depth, normals, 3DMM shape, DINO→slider bridges) provably do **not** work.

---

## 3. Section outline (with the ledger content each section draws on)

### Abstract (draft, ~150 words)
We describe a face as two orthogonal vectors: a 50-dimensional pose/expression control
built from 68 facial keypoints via 3D frontalization, and a 64-dimensional identity
carrier from an ArcFace-lineage embedding (AuraFace) compressed by supervised LDA. We
show the two are empirically orthogonal (ridge R² ≈ −0.03), that the geometry space is
identity-blind (Fisher J = 0.0847, 8.31× its noise floor) while the identity space is
near-lossless (reconstruction ceiling AUC 0.9998) and cross-shoot-discriminative
(R@1 = 0.8538). We then report, as a deliberate contribution, four **negative results**
— monocular depth, surface normals, a DINO→slider bridge, and 3DMM shape parameters all
fail to carry identity-specific geometry beyond 2D keypoints — and the gate-design
discipline (verification AUC, null controls) that caught two of our own overturned PASSes.

### 1. Introduction / problem
- Describe a photo as (a) *who* and (b) *how posed*, with a firewall between them.
- The "fast path" hypothesis: derive interpretable structural identity sliders from
  off-the-shelf monocular models. State up front that this is falsified — the paper's
  spine is the falsification plus the two survivors.
- Ledger: Phase 1 opening; the 3-partition `E = [z_g | z_d | z_a]` original design.

### 2. Data
- **FFHQ** (NVIDIA, CC BY-NC-SA 4.0) — 70k, 1 image/identity; preprocessed via
  `stratum-ffhq` (ours). One-image-per-identity **limitation stated explicitly**.
- **Proprietary multi-shoot dataset** — generic name only, ~324 personas / ~158k approved
  images, median 209 imgs/persona, multi-shoot; used **only** for cross-shoot validation,
  **never** for training. No images from it appear anywhere.
- Ledger: Phase 2 identity test set; 2026-07-20 cleaning/refit; corpus rebuild.

### 3. Method
| Signal | Carrier | Dim | Key mechanism |
|---|---|---|---|
| Geometry control | DWPose 68 iBUG keypoints → EPnP 3D frontalize → GPA → PCA → whiten | 50 | pose-invariance by construction; k=50, 99.987% variance; z_scale=1.0 |
| Identity | AuraFace 512-d → PC1+yaw removal → supervised LDA | 64 | cross-shoot R@1 = 0.8538; ceiling AUC 0.9998 |
- **The firewall:** orthogonality is *measured*, not assumed (ridge R² ≈ −0.03). Ledger Phase 5-prep.
- Block-diagonal ingestion + decoupled cross-attention (design; not yet trained) — the
  paper is about the **representations**, and says so honestly.

### 4. Results — what survived
- Masked DINOv3 patch pooling (identity, AUC 0.797, cross-shoot 0.7965). Ledger Phase 4.
- AuraFace-LDA identity: R@1 = 0.8538 cross-shoot (ledger Phase 5b + 2026-07-20 refit).
  ⚠️ *Presentation correction:* the current public post cites 0.842 (pre-refit). The
  write-up uses **0.8538** and says the basis was refit on a cleaned corpus.
- `z_g` identity-blindness: J = 0.0847 / **8.31× noise floor**, vs per-image AuraFace at
  **197.5×** — a 24× separation (ledger `zg-identity-blindness`, PARTIAL verdict).

### 5. Results — the negative results (a first-class contribution)
| Path | Verdict | Number | Ledger |
|---|---|---|---|
| z_d monocular depth | FAIL | ΔAUC −0.023 → −0.034 | Phase 2 |
| z_a surface normals | FAIL (initial PASS **overturned**) | ΔAUC −0.039 | Phase 2b |
| DINO→slider bridge | FAIL | bridge 0.704 < random-projection 0.712 ± 0.007 | Phase 3 |
| FLAME β 3DMM shape | FAIL | AUC 0.585 vs 0.69 keypoint baseline | Tier 1.1 |
| text→z_g Prior | FAIL | MSE ratio 1.75 (worse than predict-mean) | Phase 5a-exp1 |
- **The unifying claim:** monocular volumetric models hallucinate *plausible but generic*
  geometry — visually stunning, identity-poor. Do **not** retry with a better model.
- Each negative result carries its tombstone (`docs/DISCONTINUATION_NOTICE_*.md`).

### 6. Methodology contributions
- verification AUC > trace-J (trace-J is a weighted average, blind to complementarity).
- Every transfer/learned-mapping gate needs a **random-projection null** (the Phase 3b
  near-miss: the bridge beat the naive bar but lost to a random projection of its input).
- The **do-nothing null** (predict-mean / AUC 0.5) before any PASS.
- The **keypoint-resolution trap** (confidence ≠ precision; z_g 0.541 → 0.671 by crop alone).
- **Verify identity test sets visually** (the Phase 1-R contamination near-miss; the
  Phase 2b seg-collapse false PASS).
- **Re-measure at scale** (small-N Fisher-J optimism: 27 "morphology" axes → ~6; J 0.08 → 0.06).

### 7. Limitations & open debts (honest section — do not omit)
- No trained generative renderer yet; the paper describes representations, not a product.
- The AuraFace-LDA basis carries a population prior (Hegre-scaled; FFHQ spread 53%).
- `z_g` at pose-distribution extremes: encoding reliability is an open H3 question.
- Corpus-quality: extreme-pose / off-frame crops pass review and affect all three streams.
- `tainted:approved_bad_geometry` constant-distance artifact (19 rows) — unexplained.

### 8. Conclusion
Two orthogonal vectors + four falsified fast paths + the gate discipline that caught two
of our own bad PASSes.

### References
- Ledger, tree, architecture doc; FFHQ (NVIDIA); stratum-ffhq (ours); DWPose; Sapiens; EPnP;
  IP-Adapter; the literature synthesis.

---

## 4. Claims table — each headline number must trace to an artifact (governance requirement)

| Claim | Number | Ledger entry | Committed artifact |
|---|---|---|---|
| Identity cross-shoot R@1 | 0.8538 | Phase 5b / refit-cleaned | ⚠️ `experiments/geometry_pca/output/phase5b_gt_lda_refit_20260720.json` exists on disk but sits under a **gitignored** path (`.gitignore:209`) → **not evidence** by the project's rule. Producing script `46b_phase5b_gt_lda_refit.py` IS tracked. |
| z_g identity J | 0.0847 / 8.31× | `zg-identity-blindness` (PARTIAL) | `docs/assets/exp/zg-identity-blindness/fisher_metrics.json` ✅ |
| z_g high-norm KILL | 0 / 0 / 2.695 vs 1.173 | `zg-validity` | `docs/assets/exp/zg-validity/g2_diagnostics.json` ✅ |
| Normals / depth FAIL | −0.039 / −0.023 | Phase 2 / 2b | `docs/assets/exp/geometry-pca/zd_verification_auc.json` ✅ ; z_a overturn ⚠️ prose-only |
| **Masked DINO patches AUC** | **0.797** | Phase 4 | ⚠️ **No producing artifact anywhere in the repo** — verified 2026-10-01: not tracked, not under `docs/assets/`, not under `data/`. |
| DINO bridge vs null | 0.704 < 0.712 | Phase 3 | `docs/assets/exp/geometry-pca/phase3b_transfer_results.json` (⚠️ STALE verdict per ARTIFACT_STATUS) |

> ⚠️ **Blocking-for-publication finding — verified 2026-10-01 (recorded, NOT fixed):**
> by the project's own rule (*"a ledger number whose producing script no longer exists is
> not evidence"*; *"cite the TRACKED evidence path, not the gitignored `output/` copy"*),
> **three decision-critical numbers currently lack a committed producing artifact**:
>
> 1. **the headline ceiling R@1 = 0.8538** — its results JSON lives under the **gitignored**
>    `experiments/geometry_pca/output/` (`.gitignore:209`). The producing **script** IS
>    tracked, so this is a one-commit fix (copy the JSON into `docs/assets/`), not a re-run.
> 2. **Phase 4 masked-patch AUC = 0.797** — no producing artifact anywhere in the repo.
> 3. **Phase 2b face-crop overturn (−0.039)** — prose-only (already flagged in
>    `docs/assets/exp/geometry-pca/ARTIFACT_STATUS.md` → "Missing Artifacts").
>
> A scientist-facing artefact will expose all three. **Before publishing, each producing
> artifact must be located and committed into `docs/assets/` — or the claim re-derived.**
> This is a recorded blocker, not to be papered over. (Governance: `docs/02_...` records the
> same defect was fixed for `zg-identity-blindness`, whose entry cites its tracked
> `docs/assets/` path explicitly — the refit-cleaned entry's own §Artifacts still lists the
> gitignored `output/` path.)

---

## 5. Figure manifest (committed assets; **FFHQ-only** where faces are shown)

| Figure | Source asset | Public-safe? |
|---|---|---|
| ±3σ C1 / C2 traversals | `docs/assets/exp/geometry-pca/posenorm_traversal_C{1,2}.png` | ✅ FFHQ-derived (verify) |
| Scree / PCA variance | `scree.png` / `prod_scree.png` | ✅ |
| LDA axes contact sheet | must be **re-rendered from FFHQ only** (`gen_lda_axes_ffhq.py`) | ⚠️ current public one cites FFHQ IDs; reuse that workflow |
| z_g validity skeleton sheets | `docs/assets/exp/zg-validity/g1_*.jpg` | ⚠️ proprietary-derived — **do not use**; re-render or drop |

---

## 6. Public-facing constraints checklist (from the `eidolon` skill)

- [ ] Never write **"Hegre"** — always "proprietary multi-shoot dataset".
- [ ] No proprietary-dataset imagery anywhere; **FFHQ only**.
- [ ] FFHQ attributed to **NVIDIA (CC BY-NC-SA 4.0)**; `stratum-ffhq` attributed to **Tim**.
- [ ] List FFHQ image IDs used in any figure containing faces.
- [ ] Internal docs may say "Hegre"; public artefact may not.

---

## 7. What "published" means here (the one-number)

- **No trained generative model exists yet** (Phase 5 DiT unbuilt) → **no HF model card**
  is possible. The only currently-releasable HF artefact is the **already-live
  `stratum-ffhq` dataset**.
- The first artefact is therefore the **write-up itself**, published as a **lawrenz.com
  technical post** and/or an **arXiv preprint** (arXiv submission needs founder action +
  likely an endorser — a founder decision, not an agent action).
- `artefacts published` stays **0** until the live URL answers 200.

---

## 8. Release plan (in order — matches the todo)

1. ✅ Stage the outline (this document).
2. ⏳ On founder sign-off → **full draft** in `docs/briefings/` (Markdown + figures).
3. ⏳ Close the **artifact gap** in §4 (locate/commit the Phase 4 + z_a producing artifacts),
   or re-derive — a published number with no artifact violates the project's own rule.
4. ⏳ Founder approval → publish (lawrenz.com and/or arXiv); **flip public once**,
   per artefact.
5. ⏳ Verify: fetch the URL (HTTP 200 + content); record it; the one-number becomes 1.
6. ⏳ **Then resume `eidolon-publication`** (un-pause cron `6f62e66fb97b`) — it runs weekly,
   staging-only, and files a release ticket per artefact.

---

## 9. Governance notes

- The ledger is the source of truth; all numbers quoted verbatim; no plan or memory substituted.
- **Nothing marked FAIL / KILL is presented as a positive result.** The negative results are
  the paper's credibility, not hidden.
- No concluded experiment is re-run. No new experiment is started by this task.
- This document is a staged draft under the registry `artefact_dir` — it is **not** a release.
