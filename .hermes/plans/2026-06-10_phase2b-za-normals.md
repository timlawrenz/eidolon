# Phase 2b — Albedo/Surface Partition (z_a, normals) Implementation Plan

> **For Hermes:** execute task-by-task after user approval. Pre-registration (Task 0)
> MUST land in the ledger before the gate (Task 6) runs.

**Goal:** Build the surface partition `z_a` from Sapiens normal maps and prove it adds
complementary identity signal over `z_g`, gated by **verification AUC** (the canonical
instrument — trace-J is banned for concatenated partitions).

**Architecture:** Reuse the proven Phase-2 machinery (single-pass NAS cache → in-memory
IncrementalPCA fit → READ-ONLY review.db gate extract → AUC gate). The novel work is
normal-specific preprocessing (3-channel masked resample, head-pose de-rotation) and a
**representation sweep** replacing z_d's normalization sweep.

**Tech stack:** numpy, scikit-learn IncrementalPCA, sqlite3 (ro), existing
`geometry_pca` package (`pose_normalize.estimate_rotation`, `verification.partition_gate`,
`zg_inference`). CPU-only.

---

## Verified facts (probed 2026-06-10, this session)

- FFHQ `normal.npy`: (1024,1024,3) f16; **unit-norm on fg (μ=1.0000, σ=0.0001)**;
  background = zero vectors (‖n‖≈0); load ~85–105 ms/sample (NAS, fscache-warm).
- hegre `normal.npy`: variable shape (e.g. 1216×832×3); same unit-norm convention;
  mean fg normal z-dominant (+0.75) = camera-facing; range exactly [−1,1].
- z_g verification AUC baseline on the gate set: **0.5410** (seed spread 0.538–0.543,
  ±0.0025 → ε=0.01 ≈ 4σ of seed noise).
- Gate set (review.db snapshot): 102 clean identities / 1,665 encodable images.
  Growing in the background (Tim's validation window owns review.db — READ-ONLY here).

## Design decisions (and why)

1. **No normalization sweep.** Unit vectors have no scale — the affine/camera-distance
   ambiguity that killed z_d cannot exist in this representation. (This is the pivot
   thesis; the probe confirms normals are exactly unit-length.)
2. **Representation sweep instead** (the open question normals DO have):
   | Variant | Channels | Rationale |
   |---|---|---|
   | `raw` | (nx,ny,nz) 64×64×3 = 12,288-d | naive baseline — gate it first |
   | `xy` | (nx,ny) 64×64×2 = 8,192-d | nz=√(1−nx²−ny²) is redundant for camera-facing surfaces |
   | `rot` | Rᵀ·n, 3ch | **head-pose de-rotation** — normals in canonical head frame; pose is to normals what camera-distance was to depth |
   | `rot_xy` | Rᵀ·n, xy only | both corrections |
   The tangent-space log-map from the old tree note is deferred — these four are
   cheaper and likely sufficient; log-map only if all four fail marginally.
3. **One cache, variants derived at fit time.** Vector rotation is pointwise and
   linear, so it **commutes with average-pooling**: cache the resampled RAW normals
   once (69,839 × 64×64×3 float32 ≈ **3.4 GB** on NAS) + per-sample R (3,3) from
   `estimate_rotation`; derive all 4 variants in RAM (128 GB box). One NAS pass total.
4. **Pooled vectors are NOT renormalized.** Average-pooled unit vectors have ‖n‖<1
   where normals disagree locally — that magnitude IS curvature information. Keep it.
   (Renormalization = fallback sub-variant only if gating is marginal.)
5. **k=50, whitened** — consistent with the E = [z_g|z_d|z_a] partition architecture.
6. **Do NOT select a variant by retained variance** (the z_d lesson — variance rewards
   nuisance capture). Fit all, gate all, select by AUC delta.
7. **Same face-bbox crop as z_d** (`face_bbox_px`, pad=0.35) — controlled comparison.

## Pre-registered gate (Task 0 — lands in the ledger BEFORE the gate runs)

> **PASS criterion:** mean over seeds {0,1,2}: `AUC([z_g | z_a]) > AUC(z_g) + 0.01`
> on the hegre verification test (cosine, balanced same/diff pairs, n=40k pairs).
> ε=0.01 ≈ 4× the measured seed noise (±0.0025).
> Secondary report (not pass/fail): z_a-ALONE AUC — if z_a alone ≫ 0.54, normals are
> a stronger standalone carrier than geometry, which matters for E's architecture.

---

### Task 0: Pre-register the gate in the ledger
**Files:** `docs/02_EXPERIMENTS_AND_RESULTS.md`, `docs/03_EXPERIMENT_TREE.md`
Add `[Phase 2b] z_a (normals) — [ACTIVE]` section: the 4-variant sweep, the AUC gate
above, the no-normalization-sweep rationale. Commit `docs(phase2b): pre-register z_a gate`.

### Task 1: Normal preprocessing module (TDD)
**Files:** `geometry_pca/normal_encoder.py`, `tests/test_normal_encoder.py`
- `load_normal_sample(sid)` → (normal HxWx3 f32, seg, face68)
- `resample_masked_3ch(arr, x0,y0,x1,y1, out_res=64)` — per-channel NaN-aware
  average-pool (reuse `resample_masked` per channel; background→NaN via ‖n‖<0.1 ∧ seg==0)
- `head_rotation(face68_2d)` → R via `pose_normalize.estimate_rotation` against the
  production canonical template (mind the y-flip convention from `zg_inference`)
- `derive_variant(grid64, R, variant)` → flattened f32 vector (raw/xy/rot/rot_xy)
**Tests (RED first):**
1. unit-norm preserved under `rot` before pooling (‖Rᵀn‖=‖n‖)
2. shape checks: raw→12288, xy→8192
3. NaN background excluded from pooling (synthetic half-masked grid)
4. **frontalization invariant:** for a synthetically yawed normal field, mean nz(rot) > mean nz(raw)
Commit: `feat(phase2b): normal preprocessing module (TDD)`.

### Task 2: Single-pass normal cache builder
**Files:** `scripts/24_build_normal_cache_singlepass.py`
One NAS pass over ≤70k FFHQ: load normal+seg+pose once → fg-mask → face-bbox crop →
64×64×3 masked resample → store raw grid + R. Idempotent (skip if exists), progress
every 5k, **storage-rule compliant** (`data/normal_cache/` on NAS).
Output: `ffhq_normal_raw.npy` (~3.4 GB), `rotations.npy` (69839×3×3), `ids.json`.
**Cost estimate (honest):** normal.npy is 3× depth's bytes; measured ~100ms/sample →
**~2–3 h single NAS job**. Run `background=true, notify_on_complete=true`, NOTHING
else NAS-heavy in parallel (plan-skill discipline).
⚠️ **Launch requires explicit user go-ahead** (long background job — house rule).

### Task 3: Fit z_a encoders (4 variants)
**Files:** `scripts/25_fit_za_encoders.py`
Load cache into RAM (3.4 GB), derive each variant, IncrementalPCA k=50 batch=5000,
whitening stats (the `18_fit_zd_encoders.py` `_fit_from_array` pattern).
Output: `output/encoder_za_{raw,xy,rot,rot_xy}.npz` + `za_fit_summary.json`.
Cost: 12,288-d ≈ 3× z_d's 4096-d per batch → ~35–40 min/variant; 4 variants ≈ 2–2.5 h
→ background + notify. Record retained variance but **do not select on it**.

### Task 4: z_a inference helper (TDD)
**Files:** `geometry_pca/za_inference.py`, extend `tests/test_normal_encoder.py`
`encode_za(normal, seg, face, encoder, variant)` mirroring `encode_zd` (in-memory
arrays → preprocess → variant → project → whiten). Test: whitening sanity + finite.

### Task 5: Gate extractor (READ-ONLY review.db)
**Files:** `scripts/26_extract_za_gate.py`
Same query/skip logic as `20_extract_zd_gate.py` (conf≥0.5, contamination-excluded);
encode z_g once + z_a per variant; save `data/za_gate_{variant}.npz`.
Dry-run `--limit 3` first; then full run (~few minutes, 1.7k images).

### Task 6: AUC gate + nuisance audit
**Files:** `scripts/27_za_gate.py`
Per variant: `partition_gate(X_g, X_a, y, eps=0.01)` at seeds {0,1,2} → mean delta;
z_a-ALONE AUC; plus the **nuisance audit** folded in: corr(z_a top-5 components vs
estimated yaw/pitch from R, and vs fg_fraction) — the analog of the z_d C1 audit,
BEFORE trusting any PASS. Output `data/za_gate_results.json` + verdict table.

### Task 7: Record verdict in ledger (PASS or FAIL — either is publishable)
**Files:** `docs/02_…md`, `docs/03_…md`; re-index to OpenViking; push + tag.
If PASS: z_a earns its partition; next = Phase 3 DINOv3 bridge.
If FAIL: record honestly; next-hypothesis menu (tangent-space log-map, higher res,
hegre-fit control) — but note z_d precedent: don't chase marginal rescues blindly.

---

## Files created/changed
| Path | Action |
|---|---|
| `geometry_pca/normal_encoder.py`, `geometry_pca/za_inference.py` | create |
| `tests/test_normal_encoder.py` | create |
| `scripts/24_build_normal_cache_singlepass.py` … `27_za_gate.py` | create (next free numbers) |
| `output/encoder_za_*.npz`, `za_fit_summary.json` | generated (gitignored) |
| `data/normal_cache/*`, `data/za_gate_*` | generated (NAS, gitignored) |
| `docs/02_…`, `docs/03_…` | pre-registration + verdict |

## Validation
- Unit tests before each commit (`pytest tests/ -q` must stay green; currently 18).
- Dry-run extractor before full extract.
- Gate at 3 seeds; nuisance audit before accepting a PASS.

## Risks / open questions
1. **Sapiens normal quality** — predicted normals may be smoothed/generic, washing out
   identity-bearing micro-curvature. This is exactly what the gate measures; no pre-judgment.
2. **R convention mismatch** (y-flip etc.) for the `rot` variants — covered by the
   Task-1 frontalization invariant test before anything expensive runs.
3. **FFHQ→hegre basis shift** — verification AUC re-standardizes scores internally;
   a hegre-fit control encoder is the marginal-result diagnostic (as in z_d, only if needed).
4. **Lighting bias in predictions** — normals are geometry, but the *predictor* saw
   lighting; if Sapiens biases normals toward illumination, the nuisance audit
   (corr vs yaw/pitch) plus an optional lighting proxy will flag it.
5. **Cache build window** — needs Tim's explicit go (2–3 h NAS job) and coordination
   with the validation window's NAS usage (review UI reads full-res JPEGs).

## Open decisions for Tim (before execution)
1. Approve the 4-variant set (raw / xy / rot / rot_xy)? Tangent-space log-map deferred?
2. Confirm ε = 0.01 and the 3-seed mean protocol.
3. Confirm k = 50 (partition-size consistency with E).
4. When to launch the Task-2 cache build (2–3 h NAS job, background+notify)?
