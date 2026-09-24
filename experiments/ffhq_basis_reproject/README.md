# ffhq-basis-reproject

**Hypothesis:** `ffhq/stratum/{id}/auraface_lda.npy` holds **per-image AuraFace-LDA
coordinates in the PRE-refit basis** — the files are dated 2026-06-30, the pooled
basis was refit 2026-07-23, and the files match the preserved pre-refit artifact
`auraface_lda.npz.bak-20260720` exactly. Reprojecting them onto the refit basis
and L2-normalizing makes the FFHQ identity stream **encoding-identical to
`hegre_corpus`**, which is the dataset an `eidolon`-adapter DiT actually consumes.

**Differs from baseline (`refit-cleaned`):** that arm refit the basis and rebuilt
the hegre corpus, but never reprojected FFHQ. Consequence: every Eidolon-adapter
arm whose `stratum_dirs` included FFHQ (weight 2.3) trained on a 64-d identity
slot receiving **two incompatible encodings** — FFHQ pre-refit (norm ≈ 0.35),
hegre refit (norm = 1.0). No training happens in this arm; it repairs data.

**Expected outcome:** 69,960 files rewritten to the refit basis, L2-normalized;
0 errors; a byte-level backup preserving the old basis; a `BASIS_FINGERPRINT.json`
stamp enabling loaders to refuse mixed-basis inputs.

## Why this is not cosmetic

`prx-tg/production/data_stratum.py` loads the identity vector directly:

```python
identity_emb = np.load(d / 'auraface_lda.npy')   # (64,) float64
```

There is no basis check. A wrong basis is not a degraded input — it is an
unrelated 64-d vector that happens to share a filename. The five Eidolon arms on
the `exp/eidolon-conditioning` branch (eidolon-conditioning, hegre-geometry,
Arm N, Arm O) were all trained that way, so their identity conditioning is
confounded and no identity-binding conclusion can be drawn from them.

## Two different fingerprints — do not confuse

| Name | Hashed over | Describes |
|---|---|---|
| `basis_fingerprint` (new, this arm) | the basis artifacts `auraface_lda.npz` + `auraface_preprocess.npz` | the **projection** |
| `lda_basis_fingerprint` (in a corpus `_manifest.json`) | that corpus's `averages/*.lda.npy` | one corpus's **content** |

## Target convention, and why

**Refit basis + L2-normalize (norm 1.0).** `hegre_corpus` stores L2-normalized
vectors (norm exactly 1.000000). L2 normalization provably does not change cosine
geometry — measured between-image cosine is identical before and after
(0.9945 ± 0.0012) — so normalizing removes the magnitude mismatch at zero
information cost.

The rejected alternative is the **raw** refit coordinates (norm ≈ 153). That is
the convention used by the per-image retrieval tree `hegre-faces/v1/lda/`
(consumed by the Review UI `af_distance` and the GT-LDA ceiling), and it is *not*
what the DiT consumes. Recorded here so the choice is reversible and auditable.

## Out of scope (checked, deliberate)

- `hegre-faces/v1/lda/` contains **2,220 stale pre-refit files** (0.75% of
  295,468). All 2,220 are `tainted:extraction_nonface` (2,219) or
  `tainted:contamination` (1) — **none approved**, so none is reachable from
  training, the corpus, or the GT-LDA ceiling. Left untouched; recorded here.
- FFHQ's 139 missing `z_g.npy` and 41 missing `auraface_lda.npy`. The
  reprojection surface is exactly 69,960 in and 69,960 out (verified: 0
  creatable, 0 unfixable), so these gaps do not block this arm. They remain an
  open item for any loader that iterates all 70,000 dirs.
- The Synology `@eaDir` entry in `ffhq/stratum` — excluded by the script and by
  the guard; must be excluded by any loader.
- **Reprojection does not fix FFHQ's per-image identity semantics.** FFHQ is
  ~1 image per identity, so even on the correct basis its identity vectors teach
  "identity vector = per-image key". This arm makes FFHQ *encoding-consistent*,
  not *suitable as an identity target*.

## Files

| File | Role |
|---|---|
| `scripts/reproject_lda_ffhq.py` | `--dry-run` / `--apply` / `--verify` |
| `tools/hegre_dataset/basis_fingerprint.py` | guard: `stamp`, `verify`, `assert_basis` |
| `tests/tools/test_basis_fingerprint.py` | 7 tests incl. the negative control |
| `experiments/geometry_pca/output/auraface_lda.npz.bak-20260720` | pre-refit basis (evidence) |
