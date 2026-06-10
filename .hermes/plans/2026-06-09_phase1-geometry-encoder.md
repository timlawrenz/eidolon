# Phase 1 — Geometry Encoder (`z_g`) Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Produce a frozen, validated geometric PCA encoder that turns a `stratum-ffhq`
`pose.npy` into a whitened geometry slider vector `z_g`, with a scree/traversal
validation gate proving the components encode morphology — not camera pose.

**Architecture:** Deterministic linear algebra, no training. Load 68 iBUG facial
keypoints (COCO-WholeBody indices 23–90) → GPA alignment → PCA fit → whitening
stats → freeze. Validate via reconstruction error, scree curve, and ±3σ traversal
plots. See `docs/architecture.md` §3 for the full spec.

**Tech Stack:** Python, NumPy, scikit-learn (PCA), Matplotlib (validation plots),
pytest. SciPy optional (Procrustes reference).

---

## Ground Truth (verified on disk, 2026-06-09)

- Root: `/mnt/nas-ai-models/training-data/ffhq/stratum/`
- Layout: **70,000 sample directories** `00000` … `69999` (one dir per image).
  This is the *unpacked* form — NOT the tar-archive form described in the HF
  dataset card. Each dir contains the `.npy` files directly + `metadata.json`.
- `pose.npy`: shape `(133, 3)`, dtype `float16`, COCO-WholeBody, coords in `[-1, 1]`.
- Other layers per dir: `depth.npy (1024,1024)`, `normal.npy (1024,1024,3)`,
  `seg.npy (1024,1024) u8`, `dinov3_cls.npy (1024,)`, `pixel.npy (3,1024,1024)`,
  `t5_hidden.npy`, `t5_mask.npy`, `caption.txt`, `metadata.json`.
- **Branch:** Work happens on `exp/geometry-pca` (per research-workflow.md §2).
  `main` only gets the encoder if/when it graduates to core infra.

---

## Layout to create

```
experiments/geometry_pca/
├── README.md               # how to run, what each artifact is
├── requirements.txt         # numpy, scikit-learn, matplotlib, pytest, tqdm
├── geometry_pca/
│   ├── __init__.py
│   ├── constants.py        # FACE_SLICE = slice(23, 91), N_FACE_PTS = 68, etc.
│   ├── loader.py           # stream pose.npy → (68,2) arrays
│   ├── gpa.py              # Generalized Procrustes Analysis
│   ├── fit.py              # build M, fit PCA, save encoder + whitening stats
│   ├── encode.py           # project + whiten a single pose → z_g (inference)
│   └── validate.py         # scree, recon error, ±3σ traversal plots
├── scripts/
│   ├── 01_fit_encoder.py   # CLI: fit on N samples, write artifacts to output/
│   └── 02_validate.py      # CLI: load encoder, emit plots + metrics
├── tests/
│   ├── test_loader.py
│   ├── test_gpa.py
│   └── test_fit_encode.py
├── output/                 # encoder.npz, scree.png, traversal_*.png, metrics.json (gitignored)
└── data/                   # scratch / cached subsets (gitignored)
```

---

## Task 0: Environment setup

**Objective:** A working venv with the numerics stack (system Python 3.14 has no numpy/sklearn).
**Files:** Create `experiments/geometry_pca/requirements.txt`.

**Step 1:** Write `requirements.txt`:
```
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
tqdm>=4.66
pytest>=8.0
```
**Step 2:** Create venv + install:
```bash
cd experiments/geometry_pca
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
**Step 3:** Verify:
```bash
python -c "import numpy, sklearn, matplotlib; print('ok')"   # expect: ok
```
**Step 4:** Commit (`requirements.txt` only; `.venv/` gitignored).

---

## Task 1: Constants + repo hygiene

**Objective:** Central constants and a `.gitignore` so artifacts/venv never get committed.
**Files:** Create `geometry_pca/__init__.py`, `geometry_pca/constants.py`, append to repo `.gitignore`.

**Step 1:** `constants.py`:
```python
STRATUM_ROOT = "/mnt/nas-ai-models/training-data/ffhq/stratum"
N_SAMPLES = 70000
FACE_SLICE = slice(23, 91)   # COCO-WholeBody facial landmarks → 68 iBUG points
N_FACE_PTS = 68
POSE_DIM = N_FACE_PTS * 2    # 136 after dropping confidence channel
```
**Step 2:** Append to `experiments/geometry_pca/.gitignore` (or repo root):
```
experiments/geometry_pca/.venv/
experiments/geometry_pca/output/
experiments/geometry_pca/data/
__pycache__/
*.pyc
```
**Step 3:** Commit.

---

## Task 2: Pose loader (TDD)

**Objective:** Stream a sample dir's `pose.npy` → aligned-ready `(68, 2)` float32 array.
**Files:** `geometry_pca/loader.py`, `tests/test_loader.py`.

**Step 1 (RED):** Write `test_loader.py` asserting:
- `load_face_keypoints("00000")` returns shape `(68, 2)`, dtype `float32`.
- The confidence channel is dropped (only x,y remain).
- Coords lie within `[-1, 1]` (sanity; allow small epsilon).
- A helper `iter_sample_ids(limit=N)` yields zero-padded ids `"00000".."(N-1)"`.

**Step 2:** Run, verify failure (module/functions missing).

**Step 3 (GREEN):** Implement `loader.py`:
- `load_face_keypoints(sample_id)`: `np.load(.../pose.npy)` → `[FACE_SLICE]` → `[:, :2]` → `float32`.
- `iter_sample_ids(limit)`: yield `f"{i:05d}"`.
- `load_matrix(limit, drop_low_conf=True)`: stack many, optionally screen by mean
  confidence threshold to skip occluded/garbage faces.

**Step 4:** Run tests, verify pass.
**Step 5:** Commit.

---

## Task 3: Generalized Procrustes Analysis (TDD) — the load-bearing step

**Objective:** Remove translation, scale, and in-plane rotation so PCA captures
morphology, not camera mechanics. Per architecture.md §3.2.
**Files:** `geometry_pca/gpa.py`, `tests/test_gpa.py`.

**Step 1 (RED):** Write `test_gpa.py` with synthetic invariance tests:
- Take a base `(68,2)` shape. Apply a known translation+rotation+scale → aligning
  both back to the same reference yields near-identical aligned coords (atol ~1e-4).
- `gpa_align(M)` on `(K,68,2)` returns same shape; per-sample centroid ≈ 0 and
  RMS scale ≈ 1 after alignment.
- Mean shape is stable across 2 iterations (convergence).

**Step 2:** Run, verify failure.

**Step 3 (GREEN):** Implement iterative GPA:
1. Center each shape (subtract centroid).
2. Scale each to unit RMS / Frobenius norm.
3. Pick initial reference (first sample). Iterate: rotate each shape to reference
   via the orthogonal Procrustes solution (SVD of cross-covariance), recompute mean
   shape, re-normalize mean, repeat until mean-shape delta < tol (typically 2–5 iters).
- **Anchor note:** architecture.md §3.2 specifies origin at the eye-center; verify the
  iterative mean-shape result is consistent with an eye-anchored frame, or add an
  explicit eye-center pre-translation. Capture the decision in the experiment ledger.

**Step 4:** Run tests, verify pass.
**Step 5:** Commit.

---

## Task 4: Fit + freeze encoder (TDD)

**Objective:** Build `M ∈ ℝ^(N×136)` from GPA-aligned shapes, fit PCA, persist
components + whitening stats (μ_i, σ_i) + mean shape as the frozen encoder.
**Files:** `geometry_pca/fit.py`, `tests/test_fit_encode.py` (fit half).

**Step 1 (RED):** Test on a small synthetic/sampled `M`:
- `fit_encoder(M, k=50)` returns an object/dict with `components (k,136)`,
  `explained_variance_ratio (k,)`, `mean (136,)`, `whiten_mu (k,)`, `whiten_sigma (k,)`.
- `save_encoder/load_encoder` round-trips via `.npz` losslessly.
- `cumsum(explained_variance_ratio)` is monotonic and ≤ 1.

**Step 2:** Run, verify failure.
**Step 3 (GREEN):** Implement with `sklearn.decomposition.PCA`:
- Flatten aligned `(N,68,2)` → `(N,136)`; fit `PCA(n_components=k)`.
- `whiten_mu/sigma` = mean/std of the **projected** training scores per component
  (so inference whitening matches training distribution; architecture.md §7.4).
- Save to `output/encoder.npz`.

**Step 4:** Run tests, verify pass.
**Step 5:** Commit.

---

## Task 5: Inference encode (TDD)

**Objective:** `encode_pose(pose_or_id, encoder) → z_g ∈ ℝ^k` whitened, in milliseconds.
**Files:** `geometry_pca/encode.py`, finish `tests/test_fit_encode.py`.

**Step 1 (RED):** Test:
- Encoding a sample that was in the fit set yields a finite `(k,)` vector.
- Whitened training-set scores have ≈ zero mean / unit std per component (atol loose).
- End-to-end: `load_face_keypoints → gpa_align(single, to=mean) → project → whiten`.

**Step 2:** Run, verify failure.
**Step 3 (GREEN):** Implement: align single shape to the stored mean (single-target
Procrustes), flatten, subtract `mean`, project onto `components`, whiten with
`(score - whiten_mu)/whiten_sigma`.
**Step 4:** Run tests, verify pass.
**Step 5:** Commit.

---

## Task 6: Fit script (CLI)

**Objective:** One command fits the encoder on a configurable sample count.
**Files:** `scripts/01_fit_encoder.py`.

**Step 1:** Implement argparse CLI: `--limit` (default 10000), `--k` (default 50),
`--out output/encoder.npz`. Streams via loader, runs GPA, fits, saves. Prints
total variance retained at `k`.
**Step 2:** Smoke run on a small subset:
```bash
python scripts/01_fit_encoder.py --limit 2000 --k 50
# expect: "Retained X% variance at k=50" and output/encoder.npz written
```
**Step 3:** Commit script (not the artifact — gitignored).

---

## Task 7: Validation gate (the go/no-go) — architecture.md §3.4

**Objective:** Produce the empirical evidence that GPA worked and components are morphological.
**Files:** `geometry_pca/validate.py`, `scripts/02_validate.py`.

**Step 1:** Implement in `validate.py`:
- `scree_plot(encoder, out)`: cumulative explained-variance curve; mark 99% line + chosen k.
- `recon_error(encoder, M, ks)`: mean per-point reconstruction error vs. K; plot.
- `traversal_plot(encoder, comp_idx, out)`: scatter the mean shape traversed at
  −3σ … +3σ along components `C_1…C_5`; overlay to eyeball morphology vs. residual pose.
- `metrics.json`: variance@k, recon error@k, and a flag if `C_1` correlates with a
  synthetic global-rotation probe (early-warning that GPA leaked pose).
**Step 2:** `scripts/02_validate.py`: load encoder + a held-out sample of poses,
write all plots + `metrics.json` to `output/`.
**Step 3:** Run:
```bash
python scripts/02_validate.py
# expect: output/scree.png, output/recon_error.png, output/traversal_C1..C5.png, output/metrics.json
```
**Step 4:** **HUMAN GATE:** Visually inspect `traversal_C1..C5.png`. C1–C5 must read as
jaw width / face aspect / eye distance etc. — NOT global tilt/zoom. If C1 is still
camera pose → GPA failed → return to Task 3. Document the verdict in the ledger.
**Step 5:** Commit script + `validate.py`.

---

## Task 8: Preserve evidence + document (research-workflow.md §3, §4)

**Objective:** Make the result permanent and discoverable.
**Files:** `docs/02_EXPERIMENTS_AND_RESULTS.md`, `docs/03_EXPERIMENT_TREE.md`,
`docs/assets/exp/geometry-pca/`, `experiments/geometry_pca/README.md`.

**Step 1:** Copy the final validation plots + `metrics.json` into
`docs/assets/exp/geometry-pca/` (committed — this is the empirical proof, never gitignored).
**Step 2:** Write `docs/02_EXPERIMENTS_AND_RESULTS.md`: hypothesis, variance@k,
recon error, embedded traversal images, and the **verdict** (incl. any negative
findings, e.g. "GPA needed eye-anchor pre-translation, C1 leaked roll without it").
**Step 3:** Write `docs/03_EXPERIMENT_TREE.md` with status tags — Phase 1
`[ACTIVE]→[CONCLUDED]`, Phase 2 volumetric `[NEXT]`, Phase 3 DINOv3 bridge `[TBD]`,
Phase 4 DiT `[TBD]`, linking the `exp/geometry-pca` branch.
**Step 4:** Write `experiments/geometry_pca/README.md`: how to run fit + validate.
**Step 5:** Commit.

---

## Tests / Validation summary (run BEFORE each commit)

```bash
cd experiments/geometry_pca && source .venv/bin/activate
python -m pytest tests/ -q          # unit + invariance tests
```
- Prefer **invariant** tests over change-detectors (per AGENTS.md): assert GPA
  removes translation/scale/rotation and that variance ratios are monotonic — NOT
  exact eigenvalue snapshots.
- Final human gate is the traversal visual inspection in Task 7.

## Files likely to change
- New: everything under `experiments/geometry_pca/`.
- New: `docs/02_EXPERIMENTS_AND_RESULTS.md`, `docs/03_EXPERIMENT_TREE.md`,
  `docs/assets/exp/geometry-pca/*`.
- Possibly: repo-root `.gitignore`.
- `docs/architecture.md` is the source of truth — referenced, not modified, unless
  Task 7 forces a design correction (then update §3 and note it in the ledger).

## Risks / tradeoffs / open questions
1. **GPA correctness is the whole ballgame.** If C1 stays camera pose, Phase 1 fails
   its gate. Mitigated by synthetic invariance tests (Task 3) + visual gate (Task 7).
2. **Eye-anchor vs. iterative-mean reference frame** — architecture.md §3.2 says
   eye-center origin; iterative GPA converges to a mean-shape frame. Resolve in Task 3
   and record the decision. (Open question.)
3. **Low-confidence / occluded faces** pollute M. Mitigated by the confidence screen
   in Task 2 (we drop the conf *channel* for geometry but can still use it to *filter*).
4. **NAS I/O is slow** (recursive `find` over 70k dirs timed out). Loader streams by
   id and caches a subset to `data/`; fit defaults to a 10k subsample, scaled up after
   the gate passes.
5. **Scale of fit:** full 70k × 136 is tiny for in-RAM PCA — no incremental SVD needed
   here (that's a Phase 2 concern for the dense depth/normal maps).

## Execution handoff
Plan complete. Recommended execution: subagent-driven-development — a fresh
`delegate_task` per task with two-stage review (spec compliance, then code quality),
on branch `exp/geometry-pca`.
