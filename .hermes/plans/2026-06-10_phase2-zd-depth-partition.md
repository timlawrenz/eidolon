# Phase 2 — Depth Partition (z_d) Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement task-by-task once approved.

**Goal:** Build the frozen depth encoder `z_d` and prove it carries identity signal
*complementary* to 2D geometry, via the pre-registered gate
`J([z_g | z_d]) > J(z_g) × 1.15` on the reviewed hegre identity set.

**Architecture:** The depth preprocessing (`geometry_pca/depth_encoder.py`), the
IncrementalPCA fit (`scripts/18_fit_zd_encoders.py`), and the single-pass NAS cache
(`scripts/19_build_depth_cache_singlepass.py`) already exist. The missing piece is
the **gate**: a `review.db`-driven extractor that reads enriched `depth.npy` for the
89 clean identities and produces both `z_g` and `z_d` per image, then a Fisher
S_B/S_W harness that computes `J(z_g)`, `J(z_d)`, and `J([z_g|z_d])` and checks ×1.15.

**Tech stack:** numpy, scikit-learn (IncrementalPCA), sqlite3, existing
`geometry_pca` package. CPU-only. No new heavy deps.

---

## Current context / assumptions (verified on disk 2026-06-10)

- **z_g encoder shipped & frozen:** `output/encoder_production.npz` (3D-frontalize
  z_scale=1.0 → light 2D GPA → PCA → whiten; conf≥0.5 mean-per-face prefilter; k=50;
  69,851 faces). The inference path is the `encode()` closure pattern in
  `07_gate_sweep.py` lines 111–114.
- **Depth encoder built:** `geometry_pca/depth_encoder.py` — 3 modes
  (A = masked per-image z-score; A_prime = center-only, fixed `DATASET_SIGMA=0.15`;
  C = anatomical anchor, nose-relative / inter-ocular-fraction). Output is a
  64×64 NaN-aware masked resample, flattened to ℝ^4096.
- **Fit + cache built:** `18_fit_zd_encoders.py` (writes `output/encoder_zd_{A,A_prime,C}.npz`
  + `zd_fit_summary.json`), `19_build_depth_cache_singlepass.py` (writes
  `data/depth_cache/ffhq_depth_{A,A_prime,C}.npy` + `ids.json`).
- **Storage rule:** all caches/artifacts under `data/` (NAS symlink) or `output/`.
  Never local disk. `data/` → `/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data`.
- **Gate set:** `data/review.db`, `status='approved'`, excluding any persona with
  ≥1 `tainted:contamination`. Snapshot: 89 clean identities / 1,524 images. Growing.

### The data-source gap (why a NEW gate extractor is required)
The legacy `06_extract_hegre_gate.py` is **unusable for Phase 2 as-is**:
1. It uses a **hardcoded 12-identity `PICKS` list**, not `review.db`.
2. It runs **DWPose on raw JPEGs at gate time** — geometry only, no depth.
3. It predates the review system entirely.
The Phase 2 gate must instead read the **already-enriched** `depth.npy` + `pose.npy`
+ `seg.npy` from each approved image's `enriched_dir` (per `review.db`), so geometry
and depth come from the *same* enriched tensors and identities are the *verified*
clean set.

### Key risk — FFHQ→hegre distribution shift (MUST be surfaced, not hidden)
The `z_d` encoder is fit on **FFHQ** depth (tight headshots); the gate images are
**hegre** (3000–14000px editorial, often full-body crops). Sapiens depth scale and
face-bbox proportions may differ. If `z_d` fails the gate, distribution shift is a
prime suspect *before* concluding "depth carries no identity signal." The plan adds
an explicit diagnostic (Task 7) to distinguish the two.

---

## Proposed approach (ordered, honest-gate-first)

The plan runs the **encoder fit** and **cache** first (cheap, deterministic), then
builds the **gate harness**, then runs it. Per the gate-integrity rule, no free
hyperparameter (the A/A_prime/C mode choice) is locked until the honest gate picks it.

---

### Task 1: Fix the `18_fit_zd_encoders.py` docstring lie
**Objective:** Doc says "prefers local cache"; it reads the NAS symlink. One-line honesty fix.
**Files:** Modify `scripts/18_fit_zd_encoders.py`
**Step 1:** Change docstring line 4 "prefers local cache when available" →
"prefers the NAS depth cache (data/ symlink) when available".
**Step 2:** `python -c "import ast; ast.parse(open('scripts/18_fit_zd_encoders.py').read())"` → no error.
**Step 3:** Commit `cleanup: correct z_d fit docstring (cache is NAS, not local)`.

### Task 2: Build the depth cache (single NAS pass)
**Objective:** Materialize `data/depth_cache/ffhq_depth_{A,A_prime,C}.npy` so the fit is fast & repeatable.
**Files:** Run `scripts/19_build_depth_cache_singlepass.py` (no code change expected).
**Step 1:** `.venv/bin/python scripts/19_build_depth_cache_singlepass.py`
**Step 2:** Expect ~44 min for 70k (≈38ms/sample, one pass). Verify output:
  three `.npy` of shape `(N, 4096)` + `ids.json`. Confirm `N` is sane (~60k+ after
  foreground/mode filtering) and files landed under `data/` (NAS).
**Step 3:** `du -sh data/depth_cache/` — sanity-check size (3×N×4096×4 bytes).
**Step 4:** No commit (artifact is on NAS, not in git).
> **Discipline:** This is one long I/O job. Do NOT start any other NAS-heavy job
> in parallel (plan-skill pitfall). Run it alone; report when done.

### Task 3: Fit the three z_d encoders
**Objective:** Produce `output/encoder_zd_{A,A_prime,C}.npz` + `zd_fit_summary.json`.
**Files:** Run `scripts/18_fit_zd_encoders.py` (uses the Task-2 cache).
**Step 1:** `.venv/bin/python scripts/18_fit_zd_encoders.py`
**Step 2:** Expect each mode to fit in seconds-to-minutes from cache. Verify each
  `.npz` has `components (50,4096)`, `pca_mean (4096,)`, `whiten_mu/sigma (50,)`.
**Step 3:** Inspect `zd_fit_summary.json` retained-variance per mode. **Note:** depth
  is a dense 4096-dim signal — do NOT expect 99% at k=50 like geometry; record the
  actual numbers. Low retained variance is itself a finding, not a failure.
**Step 4:** Commit the summary JSON only (encoders are large; confirm with user
  whether `.npz` belongs in git or stays on NAS — see Open Questions).

### Task 4: z_d inference helper in the package
**Objective:** A reusable `encode_zd(depth, seg, face, encoder, mode)` mirroring the
geometry `encode()` closure, so the gate and future inference share one code path.
**Files:** Create `geometry_pca/zd_inference.py`; Test `tests/test_zd_inference.py`
**Step 1 (failing test):** assert that encoding a synthetic depth map through a
  toy encoder dict returns a whitened ℝ^50 vector with ~unit per-component std on a
  batch (whitening sanity).
**Step 2:** Run test → fails (module absent).
**Step 3:** Implement: reuse `depth_encoder.encode_depth_sample`-style preprocessing
  but operating on in-memory arrays (the gate already has the tensors), then
  `((vec - pca_mean) @ components.T - whiten_mu) / whiten_sigma`.
**Step 4:** Run test → passes.
**Step 5:** Commit `feat(phase2): z_d inference helper`.

### Task 5: review.db-driven gate extractor
**Objective:** For each `approved` image in clean identities, read enriched
`depth/seg/pose`, compute `z_g` (frozen production encoder) AND `z_d` (each mode),
cache to `data/zd_gate_{mode}.npz` as `X_g, X_d, y, names`.
**Files:** Create `scripts/20_extract_zd_gate.py`
**Step 1:** Query: `SELECT i.enriched_dir, p.name FROM images i JOIN personas p
  ON i.persona_id=p.id WHERE i.status='approved' AND i.persona_id NOT IN
  (SELECT persona_id FROM images WHERE status='tainted:contamination')`.
**Step 2:** Per image: load `pose.npy`→68-slice→frozen-`z_g` encode;
  load `depth/seg/pose`→`z_d` encode (per mode). Skip images whose depth fails
  foreground/mode filters (record skip count per identity — relevant to the
  documented DWPose white/black face-detection bias).
**Step 3:** Save per-mode `.npz` under `data/` (NAS). Print per-identity kept counts.
**Step 4:** Run on a SMALL `--limit 5` identities first to validate end-to-end before
  the full 89. Verify shapes: `X_g (M,50)`, `X_d (M,50)`, `y (M,)`.
**Step 5:** Commit `feat(phase2): review.db-driven z_d gate extractor`.

### Task 6: Fisher gate harness + the ×1.15 decision (TRAP-AWARE — REVISED)

> **⚠️ REVISED after the 2026-06-10 pre-gate audit (see "Audit findings" below).**
> A naive `J([z_g|z_d])` would very likely FALSE-FAIL because (1) the top z_d
> components are camera-distance/pitch nuisance and (2) hegre is out-of-distribution
> vs the FFHQ-fit whitening. The gate MUST be instrumented to separate identity
> signal from nuisance + domain shift before any verdict is trusted.

**Objective:** Compute identity-separability honestly; apply ×1.15 only to a
nuisance/shift-corrected statistic.
**Files:** Create `scripts/21_zd_gate.py`; lift `fisher_ratios` → `geometry_pca/fisher.py`.
**Step 1:** Lift `fisher_ratios` into `geometry_pca/fisher.py`; update `07_gate_sweep.py`
  to import it; quick parse/import check.
**Step 2 — re-standardize z_d on the GATE distribution (domain-shift fix):** before
  computing J, recompute per-component mean/std of z_d **on the hegre gate set itself**
  and re-whiten with those, OR z-score the concatenated vector per-component on the
  gate set. Rationale: the FFHQ whitening makes hegre land at std≈1.97; that ~2×
  inflation is pure domain shift and belongs nowhere near S_W. (Keep the FFHQ-fit
  components/basis — only re-center/re-scale the scores.)
**Step 3 — per-component Fisher J_Ci:** report `J_Ci` for every z_d component, not
  just the aggregate. This reveals which components carry identity (high J_Ci) vs
  nuisance (near-zero or scale-correlated). Print alongside the C1-nuisance
  correlations from the audit so the story is legible.
**Step 4 — top-component ablation:** compute `J([z_g | z_d])` three ways: full z_d,
  z_d minus C1, z_d minus {C1,C2}. If dropping the nuisance C1 *raises* J, that is
  direct evidence the nuisance was masking identity signal (and informs whether the
  encoder needs repair vs. the gate just needs to ignore C1).
**Step 5 — the decision:** report `J(z_g)` baseline, `J(z_d)`, and the best
  `J([z_g|z_d])` across the ablation, **per mode**, with S_B and S_W **separate**
  (degenerate-collapse guard). PASS if the corrected `J([z_g|z_d]) > J(z_g) × 1.15`.
  Pick the winning mode by corrected J **among modes that pass** — NOT by retained
  variance (the audit proved retained variance rewards nuisance capture).
**Step 6:** Write `data/zd_gate_results.json` (non-corrupt) with ALL of the above
  (baseline, per-mode, per-component J_Ci, ablation, S_B/S_W, chosen mode, and the
  re-standardization method used). Print the verdict table.
**Step 7:** Commit `feat(phase2): trap-aware z_d Fisher gate (J_Ci + ablation + shift-corrected)`.

---

## Audit findings (2026-06-10, pre-gate — drives the Task 6 revision)

Independent verification of the fitted encoders + a data-shift probe surfaced two
compounding traps. **Both are recorded in OpenViking; neither invalidates the
encoders (which are numerically clean: orthonormal, centered, no dead components).**

**Retained variance (k=50, from the fit):** Mode A 81.5%, A_prime 84.4%, C 91.4%.
→ Do NOT select on this number (see Trap 1).

**Trap 1 — nuisance-dominated top components (Phase-1 yaw trap, in depth):**
- Mode C: C1 = 40.7% of variance, correlates **r=0.969 with overall depth magnitude
  (camera distance/scale)**. The anatomical nose-anchor + inter-ocular normalization
  did NOT remove global scale.
- Mode A: C1 correlates **r=0.877 with vertical depth gradient (head pitch)**.
- ⇒ Retained variance rewards the mode that best *captures nuisance*. Mode C "wins"
  because it's most contaminated.

**Trap 2 — FFHQ→hegre distribution shift:**
- Raw mode-C depth: mean 0.539 (FFHQ) → 1.305 (hegre); std 0.598 → 1.768.
- Whitened z_d on hegre lands at per-comp |mean|=0.56, std=1.97 (should be 0/1).
- ⇒ hegre is out-of-distribution vs the FFHQ-fit whitening; the ~2× inflation
  would dump straight into Fisher S_W and mask real identity signal.

**Net:** a naive gate ⇒ probable FALSE FAIL. Task 6 is revised to be trap-aware.
Whether Mode C's normalization needs an actual *repair* (vs. the gate ignoring C1)
is deferred to the Step-4 ablation evidence — do not pre-fix the encoder.

### Task 7: Distribution-shift diagnostic (only if gate is marginal/fails)
**Objective:** Distinguish "depth has no identity signal" from "FFHQ→hegre shift
broke the encoder." 
**Files:** Create `scripts/22_zd_shift_diagnostic.py`
**Step 1:** Compare the distribution of raw `z_d` scores on FFHQ (encoder fit set) vs
  hegre gate set — per-component mean/std. Large divergence ⇒ shift suspected.
**Step 2:** Refit `z_d` on **hegre-only** approved depth (held-out split) and re-gate.
  If hegre-fit passes but FFHQ-fit fails, the signal exists; the shift is the issue.
**Step 3:** Record the finding in the ledger regardless of direction.
> Only execute this task if Task 6 is a FAIL or within ~1.05–1.20 of the bar.

### Task 8: Record the verdict in the ledger
**Objective:** Close the loop honestly.
**Files:** Modify `docs/02_EXPERIMENTS_AND_RESULTS.md` (the `[PENDING]` Phase 2 verdict),
`docs/03_EXPERIMENT_TREE.md` (status tag), re-index to OpenViking.
**Step 1:** Replace `[PENDING]` with PASS/FAIL, the actual J numbers, S_B/S_W, the
  winning mode, and the distribution-shift note. If FAIL, write it as a recorded
  negative result (the ledger's stated purpose) — do NOT bury it.
**Step 2:** If PASS: open Phase 2b (z_a / normals) as `[NEXT]`. If FAIL: document
  the next hypothesis (e.g. hegre-fit encoder, higher resolution, different mode).
**Step 3:** Commit `docs(phase2): record z_d gate verdict`.

---

## Files likely to change / be created

| Path | Action |
|------|--------|
| `scripts/18_fit_zd_encoders.py` | modify (docstring) |
| `geometry_pca/zd_inference.py` | create |
| `geometry_pca/fisher.py` | create (lift from `07`) |
| `scripts/07_gate_sweep.py` | modify (import shared fisher) |
| `scripts/20_extract_zd_gate.py` | create |
| `scripts/21_zd_gate.py` | create |
| `scripts/22_zd_shift_diagnostic.py` | create (conditional) |
| `tests/test_zd_inference.py` | create |
| `output/encoder_zd_{A,A_prime,C}.npz`, `zd_fit_summary.json` | generated |
| `data/depth_cache/*`, `data/zd_gate_*.npz`, `data/zd_gate_results.json` | generated (NAS) |
| `docs/02_…md`, `docs/03_…md` | modify (verdict) |

## Tests / validation
- Unit: `tests/test_zd_inference.py` (whitening sanity). Run with the project venv
  (`.venv/bin/python -m pytest tests/test_zd_inference.py -q`) BEFORE committing Task 4.
- Integration: Task 5 `--limit 5` dry-run before the full extract.
- The Fisher gate itself is the scientific validation; S_B/S_W reported separately.

## Risks, tradeoffs, open questions
1. **FFHQ→hegre distribution shift** (Task 7) — the headline risk. Surfaced explicitly.
2. **Gate-integrity smell test:** would a do-nothing `z_d` (all-zeros / constant)
   pass? No — a constant `z_d` adds zero between-class scatter, so `J([z_g|z_d])`
   cannot exceed `J(z_g)`. The ×1.15 bar is therefore non-trivially earned. ✓
3. **k=50 on 4096-dim depth** may retain low variance — recorded as a finding, and a
   lever (raise k) if the gate is starved.
4. **Open question — artifact storage:** do the `encoder_zd_*.npz` (could be MBs) go
   in git, or live on NAS like the caches? (Geometry's `encoder_production.npz` is
   29KB and committed; depth encoders at 4096-dim are larger.)
5. **Open question — gate set size:** run on the current 89/1,524 snapshot, or wait
   for the background review to add breadth/depth first? (Plan assumes: run now on
   the snapshot; re-runnable later.)
6. **Mode C anatomical anchor** depends on nose/eye keypoint depth lookups that can
   be noisy on extreme poses — may need a confidence guard (note for Task 5 skips).
