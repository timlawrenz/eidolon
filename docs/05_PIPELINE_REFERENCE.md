# Pipeline Reference: Eidolon Unified Dataset

> **Status:** Active — T5/caption backfill in progress (2026-06-26)
> **Branch:** `exp/text-to-zg`

## Overview

Two independent data sources feed a unified downstream pipeline that
produces DiT training artifacts: Stratum vectors (pose, depth, T5, captions),
AuraFace identity embeddings, nose-tip aligned crops, and normalized $p_g$
geometry vectors.

**Architecture principle:** The pipeline is strictly *incremental and
idempotent*. Every stage checks whether its output artifact already exists
on the NAS before doing any work. Running the same command twice is safe
and skips completed work.

---

## 1. Data Sources and Storage Locations

All heavy data lives on the NAS (`/mnt/nas-ai-models/training-data/`).
Nothing below is in the git repository (enforced by `.gitignore`).

### FFHQ (static, 70,000 images)

| Artifact                  | Path                                                           |
|---------------------------|----------------------------------------------------------------|
| Raw images                | `/mnt/nas-ai-models/training-data/ffhq/raw/*.png`              |
| Stratum output            | `/mnt/nas-ai-models/training-data/ffhq/stratum/00000/`         |
| AuraFace embeddings       | `/mnt/nas-ai-models/training-data/ffhq/auraface/*.npy`         |

### Hegre (growing, ~69,838 approved)

| Artifact                  | Path                                                                                                |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| Extracted face crops      | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/faces/<persona>/<set>/<img>`               |
| Curation database         | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db`                                 |
| Stratum output (old)      | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum/<persona>/<set>/<img>/`            |
| Stratum output (new)      | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum/faces/<persona>/<set>/<img>/`      |
| AuraFace embeddings       | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/auraface/faces/<persona>/<set>/<img>.npy`  |

> **Note:** Stratum output paths changed between the original `pose/seg/depth/normal` run
> (at `stratum/<persona>/...`) and the current `caption/t5` backfill (at
> `stratum/faces/<persona>/...`). Scripts must check both locations or use
> `rglob` when looking for artifacts.

### Deterministic pathing convention

AuraFace `.npy` files mirror the source image's relative directory
structure exactly, swapping only the extension:

```
Source:  faces/anna-l/anna-l-hegre-model/anna-l-hegre-model-01-14000px_face1.jpg
AuraFace: auraface/faces/anna-l/anna-l-hegre-model/anna-l-hegre-model-01-14000px_face1.npy
Stratum:  stratum/faces/anna-l/anna-l-hegre-model/anna-l-hegre-model-01-14000px_face1/pose.npy
```

---

## 2. Enrichment Pipeline Stages

### Stage A: Hegre Stratum Enrichment
**Tool:** `python -m tools.hegre_dataset enrich`

What it does:
1. Connects to `review.db` in read-only mode (`?mode=ro&nolock=1`)
   to avoid locking the live Review UI.
2. Queries all `status = 'approved'` images.
3. Checks which requested passes are missing by looking at the
   Stratum output directory (e.g., `t5_hidden.npy` exists? →
   skip).
4. Writes only missing images to `stratum_approved_list.txt`.
5. Invokes `stratum process` with `--device cpu` (so the 4090 stays
   free for Ollama's 27B LLM) and the delta image list.
6. After Stratum finishes, checks for missing AuraFace `.npy` files and
   extracts them using `insightface` with GPU acceleration.

**Command:**
```bash
cd ~/source/activity/eidolon
source experiments/geometry_pca/.venv/bin/activate
python -m tools.hegre_dataset enrich \
    --dataset /mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1 \
    --passes caption,t5
```

### Stage B: FFHQ AuraFace (one-shot)

**Tool:** `scripts/pipeline/extract_ffhq_auraface.py`

Run once to extract AuraFace embeddings for all 70,000 FFHQ images.
Automatically skips images that already have `.npy` files on the NAS.

**Command:**
```bash
~/.venv/bin/python scripts/pipeline/extract_ffhq_auraface.py
```

### (Future) Stage C: Nose-Tip Alignment

Will load Stratum `pose.npy` for each image, compute the eye-roll
angle, pivot on nose tip (landmark 30), save aligned crops to
`data/unified_cache/aligned/`.

### (Future) Stage D: $p_g$ Vector Extraction

Will compute normalized geometric parameter vectors from the
aligned crops and stratum pose data.

---

## 3. Current Completion Status (2026-06-26)

### FFHQ

| Pass        | Count     | Note                           |
|-------------|-----------|--------------------------------|
| Stratum     | 70,000    | All passes complete            |
| AuraFace    | 69,960    | 40 faces skipped by detector   |
| T5/Caption  | 70,000    | Complete                       |

### Hegre

| Pass        | Count     | Note                                      |
|-------------|-----------|-------------------------------------------|
| Approved    | 69,838    | Growing as review UI progresses           |
| Stratum     | 69,838    | `pose/depth/seg/normal` complete          |
| T5/Caption  | ~25,831   | Backfill in progress (~57%, ~19k remaining) |
| AuraFace    | 26,005    | Extraction in progress                    |

---

## 4. Incremental Update Workflow

When new images have been approved in the Flask Review UI
(`python -m tools.hegre_dataset review ui`) and you want to update
downstream artifacts:

### Step 1: Backfill Stratum and AuraFace
```bash
python -m tools.hegre_dataset enrich \
    --dataset /mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1 \
    --passes caption,t5
```
This is **safe to run any time**. It will:
- Skip images that already have all requested passes.
- Only invoke `stratum process` on the delta.
- Only extract AuraFace for images missing `.npy` files.

### Step 2: Refresh FFHQ AuraFace (if needed)
```bash
~/.venv/bin/python scripts/pipeline/extract_ffhq_auraface.py
```
Safe idempotent rerun — skips completed images.

### Step 3: (Future) Nose-tip alignment and $p_g$ vectors
```bash
python scripts/pipeline/sync_dataset.py  # planned
```

### Quick health check
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro&nolock=1', uri=True)
c = conn.cursor()
c.execute(\"SELECT status, COUNT(*) FROM images GROUP BY status\")
for s, n in c: print(f'{s}: {n}')
conn.close()
"
```

---

## 5. Environment Dependencies

| Component    | Requirement                                   |
|--------------|-----------------------------------------------|
| Python       | 3.14 (via `/home/tim/source/activity/eidolon/experiments/geometry_pca/.venv`) |
| cuDNN        | 9.x (installed via pip: `nvidia-cudnn-cu12`)  |
| ONNX Runtime | 1.27.0-gpu                                    |
| InsightFace  | `auraface` model (ONNX weights on NAS)        |
| Stratum-HQ   | CLI tool in venv PATH, CPU device for T5       |
| Ollama       | `gemma3:27b` for caption generation           |

### Known pitfall: cuDNN auto-detection

The venv must have `nvidia-cudnn-cu12` installed. The enrichment script
auto-injects `LD_LIBRARY_PATH` on startup via `os.execv()` restart.
No manual environment setup needed.

### Known pitfall: VRAM collision

The `gemma3:27b` caption model occupies ~20GB of the 4090's 24GB VRAM.
The enrichment script passes `--device cpu` to `stratum process` so the
T5 encoder runs on the Strix Halo CPU (128GB RAM), avoiding OOM crashes.

---

## 6. Script Inventory

| Script                                               | Purpose                              |
|------------------------------------------------------|--------------------------------------|
| `tools/hegre_dataset/enrichment.py`                  | Hegre delta enrichment (Stratum + AuraFace) |
| `tools/hegre_dataset/cli.py`                         | CLI entry point                      |
| `scripts/pipeline/extract_ffhq_auraface.py`          | One-shot FFHQ AuraFace extraction    |
| `scripts/pipeline/init_unified_ledger.py`            | NAS symlinks + SQLite ledger init    |
| `scripts/pipeline/ingest_sources.py`                 | Populate unified_state.db            |
