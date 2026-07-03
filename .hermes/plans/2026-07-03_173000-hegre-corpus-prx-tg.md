# Hegre Multi-Shot Corpus for prx-tg Identity/Geometry Disentanglement

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
**Goal:** Build a training corpus where multiple photos per persona share the same averaged AuraFace-LDA (identity) but have different z_g (geometry), forcing the DiT to disentangle the two signals rather than using z_g as an identity shortcut.
**Architecture:** The EidolonAdapter feeds AuraFace-LDA (64-d) as `global_cond` and z_g (50-d) as `sequence_cond`. The risk is that on FFHQ (1 photo per identity), z_g and AuraFace-LDA are perfectly correlated — the DiT can sidestep AuraFace-LDA entirely by treating z_g as an identity lookup. This corpus breaks that correlation by providing multiple z_g variants per single identity.
**Tech Stack:** Python 3.14, NumPy, SQLite (read-only), geometry_pca auraface_preprocessing/lda modules, PyTorch (WebDataset or stratum-format for prx-tg dataloader).

---

## Inventory: Data Availability at hegre-faces/v1

**Source DB:** `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db`
**Data root:** `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/`

| Metric | Count |
|---|---|
| Total personas | 333 |
| Personas with approved images | 325 |
| Total approved images | 140,788 |
| Personas with ≥5 approved | 323 |
| Personas with ≥20 approved | 320 |
| Personas with ≥50 approved | 317 |
| Personas with ≥100 approved | 304 |
| Max per persona | 4,130 (ariel) |
| Average per persona | 433 |

### Data Point Availability (per approved image)

| Data Point | Path Pattern | Status | Notes |
|---|---|---|---|
| **Pixels** (face crop, 1024×1024) | `faces/{persona}/{set}/{img}_face{n}.jpg` | ✅ Extracted | 140K+ crops, match prx-tg 1024px config |
| **z_g** (50-d) | `zg/faces/{persona}/{set}/{img}_face{n}.npy` | ✅ Extracted | Per-image geometry vectors |
| **AuraFace raw** (512-d) | `auraface/faces/{persona}/{set}/{img}_face{n}.npy` | ✅ Extracted | Raw AuraFace embeddings |
| **Avg z_g** (50-d) | `averages/{persona}.zg.npy` | ✅ Computed | Per-persona z_g average |
| **Avg AuraFace raw** (512-d) | `averages/{persona}.auraface.npy` | ✅ Computed | Per-persona AuraFace average (L2-normalized) |
| **AuraFace-LDA** (64-d) | *not yet extracted* | ❌ MISSING | Needs: clean_auraface() → project_to_lda() |
| **Avg AuraFace-LDA** (64-d) | *not yet extracted* | ❌ MISSING | Average of per-image LDA vectors per persona |

### Tier breakdown

| Tier | Threshold | Personas | Total Images | Description |
|---|---|---|---|---|
| **Full** | ≥20 approved, all 4 data points (pixels, raw AF, z_g, avg AF) | ~320 | ~139K | Ready after LDA projection step |
| **Enough** | ≥5 approved, all 4 data points | ~323 | ~140K | Sufficient for statistical learning |
| **Some** | ≥1 approved, has pixels + z_g, missing AF or avg | ~325 | ~141K | Partial coverage |

---

## Open Questions

1. **Contamination filtering:** Should training pairs use only `status='approved'` images from contamination-free personas (exclude any persona with ANY `tainted:contamination` image)? Or is `approved` status sufficient since the DB already separates tainted images?

2. **`tainted:approved_bad_geometry` images (59,125):** These have approved identity but failed DWPose → z_g is likely garbage. Should they be included as identity-only samples (z_g dropped/masked), excluded entirely, or given a separate z_g-is-null training mode?

3. **Corpus format:** Should the output be:
   - **(A) Stratum-style directory tree** (one dir per sample, `pixel.npy` + `identity_emb.npy` + `geometry_emb.npy` + `metadata.json`), matching `data_stratum.py` dataloader?
   - **(B) WebDataset shards** (`.tar` files with same per-sample files), matching the legacy dataloader?
   - **(C) Single monolithic `.npz`** for quick prototyping?

4. **Training integration:** Should this corpus:
   - **(A) Replace FFHQ entirely** in a new training run (pure hegre)?
   - **(B) Mix with FFHQ** in a single dataloader (joint training)?
   - **(C) Be a second-phase fine-tuning** corpus (train on FFHQ first, then fine-tune on hegre to break the correlation)?

5. **Sampling strategy:** With 304 personas having ≥100 images, should we cap the number of images per persona to prevent the largest personas from dominating? If so, what cap (e.g., 50, 100, 200)?

6. **AuraFace-LDA preprocessing:** Should we use the existing LDA basis fit on FFHQ (at `experiments/geometry_pca/output/auraface_lda.npz`) or fit a new LDA on the hegre data? The FFHQ basis was validated cross-shoot on hegre (GT-LDA kNN R@1=0.842), so it generalizes, but a hegre-native fit might be cleaner for training.

7. **Null conditioning for CFG:** The current training config uses CFG dropout categories. For the hegre corpus with AuraFace-LDA + z_g, how should CFG work?
   - Drop identity → DiT should produce a generic face with the target z_g geometry?
   - Drop geometry → DiT should produce the target identity in a neutral pose?
   - Drop both → unconditional generation?
   - Are new null embeddings needed for the hegre mode?

8. **Resolution:** The face crops are 1024px, matching the prx-tg config. Is this the resolution you want for training, or should we also produce 256px/512px variants?

9. **Validation split:** Should we hold out some personas as a validation set? If so, how many? The existing gate-validation set (120 personas from geometry_pca/review.db) overlaps partially with the hegre-faces/v1 personas — should we cross-reference to ensure no leakage?

10. **The `review.db` duality:** There are two review DBs:
    - `geometry_pca/data/review.db`: 145 personas (120-some reviewed), gate-validated
    - `hegre-faces/v1/review.db`: 333 personas, large-scale extraction pipeline
    Which should be the authoritative source for the training corpus? The v1 DB has much more data but the geometry_pca DB has stricter review (manual human review per image). Should we use v1 for volume and apply heuristics, or use geometry_pca for purity?

---

## Proposed Approach

### Phase 1: Inventory & Audit
1. Query hegre-faces/v1 review.db for contamination-free personas with ≥20 approved images
2. Verify file existence for pixels, z_g, and raw AuraFace for a random sample per persona
3. Tabulate gaps and produce an inventory report

### Phase 2: Extract AuraFace-LDA Vectors
1. Load existing `clean_auraface()` + `project_to_lda()` modules
2. For each approved image with a raw AuraFace `.npy`:
   - Load raw 512-d vector
   - `clean_auraface(v)` → remove PC1 (domain) + yaw (pose) nuisances
   - `project_to_lda(v_clean)` → project to 64-d identity space
   - Save as `lda/faces/{persona}/{set}/{img}_face{n}.npy`
3. Compute per-persona averages of the LDA vectors and save as `averages/{persona}.lda.npy`
4. All processing must be **idempotent** (skip existing outputs)

### Phase 3: Build Training Corpus
1. For each approved image in a contamination-free persona with ≥N images (N configurable, default 5):
   - Load average AuraFace-LDA for the persona → `identity_emb` (64-d)
   - Load individual z_g → `geometry_emb` (50-d)  
   - Load/resize face crop → `pixel` (3, H, W)
2. Output format: **Option A (stratum-style)** initially for quick `data_stratum.py` compatibility
3. Metadata: `{"persona": name, "set": slug, "image_id": ..., "identity_emb_shape": [64], "geometry_emb_shape": [50]}`

### Phase 4: Validation & Smoke Test
1. Verify that different images of the same persona share the same identity_emb
2. Verify that different images of the same persona have DIFFERENT geometry_emb (correlation check)
3. Smoke test with prx-tg dataloader: load a batch and verify shapes
4. Generate a few training-step samples to visually confirm identity consistency across z_g variants

### Phase 5: Training Run (separate plan)
1. Modify prx-tg config to use EidolonAdapter with hegre corpus
2. Add CFG dropout categories for identity/geometry
3. Run training and monitor identity consistency + geometry diversity metrics

---

## Files Changed

| File | Action | Purpose |
|---|---|---|
| `tools/hegre_dataset/review/geometry.py` | PATCH (+130 lines) | Added `compute_lda_vectors()` — clean AuraFace + project to LDA + per-persona averages |
| `tools/hegre_dataset/cli.py` | PATCH (+20 lines) | Registered `review compute-lda` and `build-corpus` subcommands |
| `tools/hegre_dataset/corpus_builder.py` | CREATE | `build_corpus()` — assemble training triples into stratum-style directories |
| `scripts/pipeline/hegre_corpus_inventory.py` | CREATE | Inventory/audit script (Phase 1) |
| `production/data_stratum.py` (prx-tg) | PATCH | Minimal `_load_sample_eidolon` — loads only pixel + identity + geometry, no T5/DINO/pose/seg |

## Commands

```bash
# 1. Inventory
python scripts/pipeline/hegre_corpus_inventory.py

# 2. Extract AuraFace-LDA (run this first — one-time, idempotent, resumable)
python -m tools.hegre_dataset review compute-lda \
  --dataset /mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1

# 3. Build corpus (requires Phase 2 complete for all personas)
python -m tools.hegre_dataset build-corpus \
  --dataset /mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1 \
  --output /path/to/hegre_corpus \
  --min-images 5 \
  --max-images 100

# 4. Smoke test with prx-tg dataloader
cd /path/to/prx-tg && .venv/bin/python -c "
from production.data_stratum import StratumDataset
ds = StratumDataset(stratum_dir='/path/to/hegre_corpus', batch_size=4, 
                     target_latent_size=1024, max_samples=N, adapter_name='eidolon')
batch = next(iter(ds))
print(f'identity: {batch[\"identity_emb\"].shape}, geometry: {batch[\"geometry_emb\"].shape}')
"
```

## Smoke Test Results

- Shape: `image_data` (4, 3, 1024, 1024), `identity_emb` (4, 64), `geometry_emb` (4, 50) ✓
- Same-persona identity vectors are identical ✓
- Same-persona geometry vectors are different ✓
- Dataloader infinite iteration works correctly ✓

## Next Steps (User)

1. **Run `review compute-lda`** for all 333 personas (~57K images, ~2-3 hours on CIFS NAS)
2. **Run `build-corpus`** to produce the stratum-style corpus on NAS
3. **Update prx-tg config** to use `adapter_name: eidolon` with the hegre corpus path
4. **Train** — the corpus can be mixed with FFHQ by pointing the dataloader at the hegre output directory

---

## Risks & Tradeoffs

1. **z_g quality:** z_g vectors depend on DWPose keypoint quality. `tainted:approved_bad_geometry` images may have degenerate z_g. We must filter or mask these.
2. **AuraFace quality:** AuraFace may fail on extreme profiles or occlusions. Check for degenerate (all-zero or near-zero-norm) embeddings.
3. **Identity leakage through z_g:** Even on hegre, z_g may carry residual identity signal (the gate showed J≈0.06 — small but non-zero). Monitor Fisher ratio on the corpus split.
4. **NAS I/O:** Processing 140K images from CIFS NAS will be slow. Plan for resumable, idempotent scripts with progress bars.
5. **Persona name collisions:** Suffix-aware keys (`darina-l` vs `darina`) are respected in the pipeline; verify no collisions in the output paths.

---

## Validation Gates

1. **Null model:** Shuffle identity_emb across images. Training loss should not drop (identity carries no information about pixels when geometry is fixed).
2. **Identity consistency:** Same persona + different z_g — generated faces should have the same identity.
3. **Geometry diversity:** Same identity_emb + different z_g — generated faces should have different poses/expressions.
4. **Cross-persona discrimination:** Different identity_emb + same z_g — generated faces should have different identities.
