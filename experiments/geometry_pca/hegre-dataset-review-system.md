# Hegre Dataset Review System

## Purpose

Review ~2400 images across 120 identities from the hegre dataset to build a
clean multi-identity validation set for the Fisher S_B/S_W identity-separability
gate. Each identity has ~20 images drawn from distinct editorial shoots.

**Why this exists:** The Phase 1-R identity gate (geometry encoder validation)
used only 10 identities after discovering that 4 of the original 5 were
contaminated. This review system scales to 120 identities so the volumetric
encoders (Phase 2: z_d, z_a) can be validated on a statistically robust basis.

## Architecture

```
overnight_identity_map.json   (identity labels per image)
        │
        ▼
 13_seed_review_db.py          (populates SQLite: personas, sets, images)
        │
        ▼
 14_import_review_json.py      (imports pre-existing review decisions)
        │
        ▼
 15_review_ui.py               (Flask web UI on port 5100)
        │
        ▼
   data/review.db              (SQLite — single source of truth)
```

### Database schema (`data/review.db`)

- **personas** — `id, name` (120 rows, one per model identity)
- **sets** — `id, persona_id, slug` (one per editorial shoot)
- **images** — `id, persona_id, set_id, image_path, enriched_dir, status, reviewed_at`
  - `image_path`: absolute path to original JPEG on NAS
  - `enriched_dir`: path to stratum-hq output (depth, normal, seg, pose)
  - `status`: `unreviewed`, `approved`, `tainted:contamination`, `tainted:extraction_black`, `tainted:extraction_nonface`, `tainted:insufficient`

### Status meanings

| Status | Meaning | Action at gate time |
|--------|---------|---------------------|
| `approved` | Verified single-identity, usable crop | ENCODE for gate |
| `tainted:contamination` | Different person / male face / merged identity | DROP identity entirely |
| `tainted:extraction_nonface` | Non-face crop (hair, body, background) | DROP crop only |
| `tainted:unusable` | Full-image fallback / unusable crop (current no-face fallback) | DROP crop only |
| `tainted:insufficient` | Too few usable images | DROP identity |
| `tainted:extraction_black` | **(LEGACY)** Black square — old no-face fallback emitted a black image | DROP crop only |
| `unreviewed` | Not yet looked at | SKIP (don't use in gate) |

> **Fallback-behavior change (commit `91b527f`):** stratum-hq's no-face fallback
> used to emit a **black image** (→ `tainted:extraction_black`). It now falls back
> to the **full uncropped image**, which is labeled `tainted:unusable` (or
> `tainted:extraction_nonface` for a non-face crop). Consequently `extraction_black`
> has **0 live rows** in the current DB and is retained only for historical decisions.

### Current state (2026-06-10)

- 120 personas, 2400 images — **review complete (0 unreviewed)**.
- Live breakdown: **1,589 approved**, 440 `insufficient`, 269 `extraction_nonface`,
  69 `contamination`, 33 `unusable`.
- **Gate-ready: 1,524 approved images across 89 contamination-free identities**
  (8 personas carry ≥1 `contamination` image → excluded entirely).
- The corpus is **growing in the background**: more personas (breadth) and more
  images per persona (depth) are being reviewed continuously. Treat 89/1,524 as a
  snapshot — the gate must be re-runnable as the DB expands.
- Historical Phase-1-R contaminations (legacy 10-id gate): `muriel`, `natalia-a`.

## How to use the review UI

### Start the server

```bash
cd /home/tim/source/activity/eidolon/experiments/geometry_pca
.venv/bin/python scripts/15_review_ui.py --port 5100
```

Open **http://127.0.0.1:5100** in a browser.

### Review workflow

1. A **random persona** is shown with all images in **random order**
2. Select a **brush** (Black, Non-face, Contamination) — click images to taint them
3. Click **DONE** — approves all remaining images, loads the next persona
4. Color-coded borders: green=approved, black=tainted black, red=tainted non-face, pink=tainted contamination

### Review guidelines

- **Identity contamination** (male face, different woman): use Contamination brush
- **Extraction failures** (black squares, non-face crops): use Black or Non-face brush
- Otherwise DONE approves everything — the identity is verified clean
- If you're not sure about a crop, leave it unreviewed and close the browser;
  it will show up again on next load

## Data locations

| What | Where |
|------|-------|
| Source images (NAS) | `/mnt/nas-ai-models/training-data/loras/hegre-14000px/` |
| Enriched outputs (NAS) | `/mnt/nas-ai-models/training-data/eidolon/hegre_enriched/` |
| Symlink to enriched | `experiments/geometry_pca/data/hegre_enriched → NAS` |
| Review database | `experiments/geometry_pca/data/review.db` |
| Identity labels | `experiments/geometry_pca/data/overnight_identity_map.json` |
| Pre-review collages | `experiments/geometry_pca/output/collages_120/` |
| Collage gallery HTML | `experiments/geometry_pca/output/collages_120/review.html` |

All paths under `experiments/geometry_pca/data/` resolve to NAS via symlink
(`data → /mnt/nas-ai-models/training-data/eidolon/geometry_pca_data`).

## Key scripts

Numbering reflects phase lineage after the Jun-2026 cleanup
(review block 09–17, Phase 2 z_d block 18–19):

| Script | Purpose |
|--------|---------|
| `09_plan_overnight_enrich.py` | Built the image list (120 identities × 20 images) |
| `10_verify_gate.py` | Auto-flag suspicious sets (keywords, low confidence) |
| `11_build_120_collages.py` | Generates face-crop collages for all 120 identities |
| `12_build_html_gallery.py` | Builds scrollable HTML gallery from collages |
| `13_seed_review_db.py` | Creates/rebuilds SQLite database from identity map |
| `14_import_review_json.py` | Imports review decisions from `gate_review.json` into DB |
| `15_review_ui.py` | Flask web UI for interactive review |
| `16_batch_resumable_enrich.py` | Resumable hegre enrichment runner (stratum-hq passes) |
| `17_merge_batch.py` | Merge per-batch identity maps into the review DB |
| `18_fit_zd_encoders.py` | **Phase 2:** fit z_d depth PCA encoders (3 normalization modes) |
| `19_build_depth_cache_singlepass.py` | **Phase 2:** single-pass NAS depth cache builder |

## Identity extraction rules

- **Suffix-aware keys**: `darina-l` and `darina` are DIFFERENT people
- **Couple sets excluded**: any set with `-and-` or `couple` in the slug is excluded
  from the overnight enrichment (DWPose can grab the wrong person)
- **Enrichment passes**: pose, seg, depth, normal — run via stratum-hq on the RTX 4090
  (see `data/overnight_enrich.log`)

## Known issues

- Some images produce **black squares** (DWPose YOLOX found no person)
- Some images produce **non-face crops** (DWPose hallucinated a face on hair/body)
- **Face detection bias:** DWPose/YOLOX performs noticeably better on white faces than
  black faces. Black identities will have systematically higher extraction-failure
  rates and fewer approved crops per persona. Gate should compensate with per-identity
  thresholds rather than a flat minimum.
- The `alya` collage contains one **artistic sketch** (non-photographic) —
  extraction failures like this should be tainted, not the whole identity
- Couple-set filter catches `-and-` but misses partner photos in sets
  named without the pattern (e.g. `muriel` had a male face despite no `-and-`)
- Collage generation is slow (~1 identity/sec) because it loads full 14000px JPEGs;
  all 120 collages built and reviewed as of 2026-06-10 (review complete).

## Gate usage (for the Fisher S_B/S_W test)

When building the z_d and z_a gate:
```sql
-- Get all approved images for clean identities
SELECT i.id, i.image_path, i.enriched_dir, p.name as persona
FROM images i JOIN personas p ON i.persona_id = p.id
WHERE i.status = 'approved';
```

Identities with ANY `tainted:contamination` images should be excluded entirely
from the gate, even if other images for that persona are `approved`.
