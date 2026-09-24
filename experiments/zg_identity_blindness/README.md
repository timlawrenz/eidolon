# Arm: `zg-identity-blindness` — re-measure z_g's identity content on the curated corpus

**Status:** `[PRE-REGISTERED]` — gates written 2026-09-24, **before** the first run.
**Mode:** `confirmatory` (see `provenance.yaml`; declared before run 1)
**Branch:** `exp/zg-identity-blindness`
**Base:** branch tip `8897b90` on `exp/zg-validity`'s line, **not `main`** — see §7.
**Cost:** CPU only. No GPU. No model. Reads existing per-image arrays.

---

## 1. Why this arm exists

Two independent reasons, either one sufficient:

**(a) The belief is load-bearing and its magnitude is unverified.** The Eidolon
disentanglement design rests on the claim that **`z_g` is a geometry/pose control
space that carries almost no identity**, with identity living in AuraFace. The
project is now about to spend real GPU time testing pose-vs-identity orthogonality
on prx-tg. If that premise is wrong, the GPU time answers a question about a
premise nobody re-checked.

**(b) The number's producing script does not exist.** The ledger's
`[Phase 5-prep] text-to-zg` entry (2026-06-27) records **global Fisher J = 0.059**
on the "full 69,110 / 323-identity corpus", with the axis-count collapse
"morphology axes (J>0.15): 27 → 6". Verified 2026-09-24 across **every branch**:
the only scripts consuming `geometry_pca.fisher.fisher_ratios` are
`07_gate_sweep.py`, `21_zd_gate.py`, `22_zd_complementarity_diagnostic.py` and
`32_phase3_systematic_review.py` — all z_d / legacy-gate scripts operating on the
**legacy 1,448 / 101** set. **No script on any branch computes the global J on the
corpus.** The sapiens2 scripts (`fisher_split_mixed.py`, `fisher_split_sapiens2.py`)
compute a *different* quantity (mean of per-axis J) and different numbers
(0.136 DWPose / 0.331 Sapiens2).

By this project's own rule — *"A ledger number whose producing script no longer
exists is not evidence"* (AGENTS.md) — **J = 0.059 is not currently evidence.**
This arm rebuilds the instrument in-repo.

**(c) The data changed underneath it.** 0.059 was measured on the pre-curation
corpus (**69,110 samples / 323 personas**) while `62k` were `bad_geometry` and
`216k` unreviewed — the ledger flags it *"directional, not final."* The corpus is
now **31,711 samples / 321 personas, 100% `approved`** in the review DB. The
number is being inherited from a dataset that no longer exists.

---

## 2. Hypotheses

* **H₀ (the belief under test):** `z_g` carries almost no identity. Global Fisher
  J ≪ 1, and the per-axis morphology block (J > 0.15) is small.
* **H₁ (the data-change prediction — declared in advance):** J = S_B / S_W, and the
  ledger says DWPose noise *inflates the denominator* S_W. Curating the corpus
  removes that noise, **shrinking S_W and therefore RAISING J.** A materially
  higher J means `z_g` carries more identity than believed, and the orthogonality
  premise is weaker than the design assumes.

**H₁ is the reason this is pre-registered rather than casually re-run: an increase
is the predicted outcome, and predicting it in advance is what keeps it from being
read as a surprise or rationalised after the fact.**

---

## 3. Instrument (rebuilt, and declared, because the original is lost)

| element | definition |
|---|---|
| metric | **global** Fisher J = S_B / S_W, via `geometry_pca.fisher.fisher_ratios` |
| input | per-image `z_g.npy` (50-d whitened PCA of the 68 DWPose face keypoints) |
| labels | `persona` from the corpus `metadata.json` |
| scatters | S_B = Σ_c n_c·‖μ_c − μ‖² / N ; S_W = Σ_c Σ_i ‖z_i − μ_c‖² / N ; both reported **separately** |

**Because the original "Tier 0.3" filter definition is lost with its script, the
quantity is not reproducible verbatim.** The arm therefore re-measures a declared
**venue family** and reports the spread rather than a single number:

* venue **A** — no confidence filter (all corpus samples)
* venue **B** — mean face-keypoint confidence ≥ **0.3** (the most likely reading of "Tier 0.3")
* venue **C** — mean face-keypoint confidence ≥ **0.5** (the `07_gate_sweep.py` convention)

Primary venue for the gate is **B**. Venues A and C are reported for sensitivity.

---

## 4. Pre-registered gates

Written **before** the first run. Falsifier stated for each.

### G0 — instrument identity (must pass before anything is interpretable)
Corpus `z_g` must be **bit-identical** to the encoder source on 400 sampled
samples: `max‖diff‖ = 0.000e+00`, `missing = 0`.
**FAIL → stop.** A mismatch means the input is not the real `z_g` and no downstream
number is about `z_g`.

### G1 — positive control (the anti-null check; the load-bearing gate)
Run the **identical** instrument on `auraface_lda.npy` under venue B.
**Require `J_auraface ≥ 3 × J_zg`.**

Without this, a low J for `z_g` is uninterpretable: it could mean "`z_g` is
identity-blind" (the belief) or "**this instrument cannot detect identity
separability at all**" (a broken instrument). AuraFace is known to carry identity
(the LDA basis gives R@1 = 0.8538), so it is the correct positive control. If J
fails to separate AuraFace from `z_g`, **the arm is void and reports void.**

### G2 — the headline re-measurement (primary gate)
Global `J(z_g)`, venue B, on the curated corpus, plus the per-axis morphology count
(axes with J_Ci > 0.15).

* **CONFIRM** the belief if `J ∈ [0.02, 0.12]` **AND** morphology-axis count ≤ **10**
* **FALSIFY** the belief if `J ≥ 0.20` **OR** morphology-axis count ≥ **20**
* **PARTIAL** otherwise → the belief's *magnitude* is corrected; neither
  "confirmed" nor "falsified" is written.

**Declared direction:** an increase relative to 0.059 is the predicted outcome
(§2 H₁). It is a *result*, not an anomaly.

### G3 — legacy collapse reproducibility
Attempt to reproduce the "27 → 6 morphology axes" collapse on a legacy-style
subset. **If the original's selection cannot be reconstructed, record
`UNREPRODUCIBLE` — do not invent a stand-in and call it a reproduction.**

---

## 5. Adversarial requirements (mandatory before any verdict)

1. **S_B and S_W reported separately.** A high J achieved by a *collapsed* S_B is
   not identity separability. Both are printed.
2. **Per-class sizes reported**, and identities with `< 2` samples excluded and
   counted (within-person scatter is undefined for them).
3. **The venue spread is reported in full.** Reporting only the venue that
   agrees with 0.059 is p-hacking by filter.
4. **Population check:** report sample count and persona count actually used vs
   the corpus total, and state the shortfall.
5. **The 0.059 comparison is made explicit**, including that the original was on a
   different and larger dataset.

---

## 6. What would make this arm uninformative

* G0 fails (input is not the real `z_g`).
* G1 fails (instrument cannot detect identity separability → every J is
  meaningless).
* Persona yield is so low (after the ≥2-sample floor) that within-person scatter is
  estimated from a handful of pairs → J is noise.

---

## 7. Base-branch deviation (explicit, per AGENTS.md)

`docs/00_GIT_WORKFLOW.md` rule 2 requires a fresh `exp/*` branch **from `main`**.
**This arm cannot be branched from `main`.** Verified 2026-09-24: `main` is **220
commits behind** and **lacks the dependencies this arm needs**:

| required | on `main`? |
|---|---|
| `tools/hegre_dataset/models.py` (the corpus loader) | **MISSING** |
| `tools/hegre_dataset/basis_fingerprint.py` | **MISSING** |
| `scripts/reproject_lda.py` | **MISSING** |
| `experiments/zg_validity/src/run_gates.py` | **MISSING** |
| build-corpus tooling | **MISSING** |

Branching from `main` would yield a tree that cannot even load the corpus. Per
AGENTS.md — *"If the correct base lacks a dependency the arm needs, say so
explicitly rather than silently branching from an unrelated `exp/*` branch"* — the
arm is based on the current tip (`8897b90`), which carries the data infrastructure,
and the deviation is recorded here rather than left implicit.

**This is a standing blocker, not an arm-specific one**: every future eidolon arm
hits it until `main` is merged forward.

---

## 8. Ledger obligation

On completion: `docs/02_EXPERIMENTS_AND_RESULTS.md` entry with the pre-registered
gates restated verbatim, the measured evidence, the 7-box adversarial pass, and a
verdict — plus a `docs/03_EXPERIMENT_TREE.md` marking and a provenance `notes:`
fill. Evidence artifacts (metric JSON, any plot) committed to
`docs/assets/exp/zg-identity-blindness/`.
