---
name: hegre-dataset-pipeline
description: "Instructions for using, extending, and debugging the Hegre Face Dataset extraction and review pipeline."
tags: [dataset, eidolon, mtcnn, stratum, geometry, review-ui]
---

# Hegre Face Dataset Pipeline

This skill defines the architecture and agentic workflows for the `tools.hegre_dataset` module within Project Eidolon. This pipeline discovers, extracts, filters, and geometrically verifies face crops from massive (14000px) photo shoots to create mathematically pure datasets for Eidolon.

## 1. Core Philosophy: Purity First and The z_g Vector

The dataset optimizes for **geometric purity** (favoring false negatives over false positives). 
Because Eidolon splits identity into E = [z_g | z_d | z_a] (Geometry, Micro-depth, Albedo):
* We use **MTCNN** as a dumb region proposal network (fast, high recall, low precision).
* We use **Stratum (DWPose)** and 3D-aware frontalization (z_g) as the mathematical discriminator.
* If DWPose fails on a valid face, we don't throw it away. We mark it as `tainted:approved_bad_geometry` so it can be used to train appearance (z_a) while being masked out of the geometric (z_g) loss.

## 2. Pipeline Stages

The pipeline is executed sequentially via the `python -m tools.hegre_dataset` CLI:

### Phase 1: Discovery (discover)
Scans the NAS ground truth directory.
* Uses a suffix-aware identity key (`idkey()`) so `darina-l` is treated as a separate identity from `darina`.
* Filters out identities with fewer than `--min-sets` (default 3).
* Produces `manifest.json`.

### Phase 2: Extraction (extract-faces)
Extracts square face crops from the 14000px source JPEGs.
* **OOM Prevention:** Downscales massive images to 4000px for MTCNN detection, scales bounding boxes back up, and performs the final crop on the 14000px original.
* **Crop rules:** Square, shifted/clamped to image bounds, absolutely **no zero-padding**, resized to exactly 512x512 using Lanczos downscaling (no upscaling).
* Processes identities breadth-first (round-robin) to surface diverse faces early.

### Phase 3: Database Seeding (review init / review seed)
Tracks extraction progress.
* Schema: `personas` -> `sets` -> `images`.
* Unique constraint: `(persona_id, set_id, source_image, face_index)`.

### Phase 4: Enrichment (enrich)
Generates a list of absolute paths for all `approved` faces (or `unreviewed` if running early), and natively invokes `stratum process --passes pose,seg,depth,normal`.

### Phase 5: Geometric Clustering (review split-persona and review compute-geometry)
Solves the "Mixed Persona" (co-stars) problem mathematically.
* `split-persona`: Loads all z_g vectors for a given persona and runs DBSCAN (default eps=20.0). It fractures a mixed `anna` persona into `anna_cluster_1`, `anna_cluster_2`, leaving noise behind.
* `compute-geometry`: Averages the z_g vectors of valid (unreviewed/approved) images to find the **True Identity Centroid**. Calculates and writes the Euclidean distance for every image into the `zg_distance` SQL column.

### Phase 6: Human Review (review ui)
A Flask web interface running on port 5101.
* **Worst-First Sorting**: If z_g distances are computed, the `unreviewed` queue sorts `ORDER BY zg_distance DESC NULLS LAST`, feeding the reviewer the worst outliers and non-faces first.
* **True Identity Anchors**: The UI pins the 3 images *closest* to the z_g centroid at the top of the screen to prevent reviewer drift.
* **X-Ray Mode**: Toggles `?skel=1`, drawing the 68 DWPose keypoints on the image. Dot radius scales with DWPose confidence.
* **Modality Masking Brushes**: `Non-face`, `Contamination`, `Unusable`, and `Bad Geometry`.

## 3. Agent Rules and Pitfalls

When modifying or testing this codebase, adhere strictly to these rules:

1. **Never mutate data/review.db during tests:** Tests must use a `tmp_path` SQLite DB. The live DB has active WAL connections and manual human annotations.
2. **Missing stratum Paths:** Stratum flattens its output. `faces/anna_cluster_1/anna-shoot/img1_face1.jpg` maps to `stratum/anna/anna-shoot/img1_face1/pose.npy`. Always use `.split('_cluster_')[0]` to find the base directory, and `rglob(f"{Path(img_path).stem}/pose.npy")` to find the artifact.
3. **Python 3.14 SQLite db.changes:** Do not use `db.changes` to count rows. It was removed in Python 3.14. Use `db.execute("SELECT changes()").fetchone()[0]`.
4. **Normalized Coordinates:** Stratum DWPose outputs coordinates normalized around 0 (e.g., `[-1, 1]`). To plot them on the 512px crop, map them using `px = (x / 2.0 + 0.5) * img_w`.
5. **Ignore Tainted Images in Math:** Never include images with a `tainted:` status when computing centroids or running DBSCAN. They will poison the geometric average.

## 4. Example CLI Workflow

```bash
# 1. Discover
python -m tools.hegre_dataset discover --source /mnt/nas/... --dataset data/hegre_datasets/v1

# 2. Extract (Instant resume supported via tqdm/manifest checks)
python -m tools.hegre_dataset extract-faces --dataset data/hegre_datasets/v1

# 3. DB Init and Seed
python -m tools.hegre_dataset review init --dataset data/hegre_datasets/v1
python -m tools.hegre_dataset review seed --dataset data/hegre_datasets/v1 -v

# 4. Enrich and Compute Geometry
python -m tools.hegre_dataset enrich --dataset data/hegre_datasets/v1
python -m tools.hegre_dataset review compute-geometry --dataset data/hegre_datasets/v1 --encoder experiments/geometry_pca/output/encoder_production.npz

# 5. Split Co-Stars (if needed)
python -m tools.hegre_dataset review split-persona --dataset data/hegre_datasets/v1 --persona anna --encoder ...

# 6. Review
python -m tools.hegre_dataset review ui --dataset data/hegre_datasets/v1 --port 5101 --bind 0.0.0.0
```
