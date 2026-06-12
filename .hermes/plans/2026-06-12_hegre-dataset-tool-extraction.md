# Extract Hegre Face-Dataset Tooling into Standalone Module

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task. **Execute subagents SEQUENTIALLY, never in parallel**, because the tasks are highly coupled (e.g., UI depends on DB schema).
> **Note:** This plan is designed to be executable by an agent with NO access to the background knowledge of the conversation that produced it. All context is self-contained.
> **CRITICAL DB GUARD:** Subagents MUST use a dummy/temporary directory (e.g., `/tmp/hegre_test_dataset/`) when verifying schema creation, seeding, and the UI. NEVER read, seed, or interact with the live `data/review.db` during development/testing.

**Goal:** Extract the face-dataset creation pipeline (identity discovery, MTCNN face extraction, review UI, stratum enrichment) from `experiments/geometry_pca/` into a standalone `tools/hegre_dataset/` Python package. The experiment's existing scripts and data remain frozen and untouched.

**Architecture:** Single MTCNN-based face detection pipeline that produces both the review thumbnail AND the final asset. Multi-face detection (`_face1`, `_face2`, ...) surfaces all detectable faces for honest review. Each dataset version is a self-contained directory with provenance metadata. The experiment continues to read its existing data via the unchanged `data/` symlink.

**Tech Stack:** Python 3.12+, MTCNN (facenet_pytorch), SQLite, Flask, stratum-hq CLI (external), PIL/Pillow with `MAX_IMAGE_PIXELS=None`.

---

## Design Decisions (Pre-Registered)

1. **Single detector:** MTCNN for both the review thumbnail and the asset. No DWPose-keypoint-based cropping in the review path. What the reviewer sees IS what gets enriched.
2. **Multi-face:** All detectable faces per source image are extracted as `{filename}_face1.jpg`, `{filename}_face2.jpg`, etc. Review determines which belong to the target identity.
3. **No pre-sampling:** All images from all shoots for selected identities are extracted. Review handles the volume incrementally (one persona at a time).
4. **Square crops, shift/clamp, no padding:** Follows the ViT cropping pattern from the plan skill reference. See `35_build_hegre_face_dataset.py`'s `get_square_box()` for the existing implementation — port it, do not change the algorithm.
5. **Existing experiment frozen:** The original scripts (`09_plan_overnight_enrich.py` through `17_merge_batch.py`, `35_build_hegre_face_dataset.py`) stay in `experiments/geometry_pca/scripts/`. They are annotated with a header comment: `# Superseded by tools/hegre_dataset/ as of 2026-06-12. Frozen as part of Phase 1–4 experimental record.` No files are deleted or moved.
6. **Identity suffix-awareness:** The identity key function from `09_plan_overnight_enrich.py` (suffix-aware, e.g., `darina-l` ≠ `darina`) is preserved.
7. **No couple-filter at extraction time:** The review system handles multi-model images by surfacing all faces. The `-and-`/`couple` heuristic from the old tool is not reimplemented — the reviewer catches contamination, not a regex.

---

## Output Structure

```
data/hegre_datasets/v{N}/
├── metadata.json          # Provenance: source, config, review stats
├── review.db              # SQLite (personas → sets → images with face_index)
├── faces/
│   └── {identity}/
│       └── {set_slug}/
│           ├── {original_filename}_face1.jpg
│           ├── {original_filename}_face2.jpg
│           └── ...
├── stratum/               # stratum-hq output (populated by enrichment pass)
│   └── {identity}/
│       └── {set_slug}/
│           └── {original_filename}_face1/
│               ├── pose.npy
│               ├── depth.npy
│               ├── normal.npy
│               ├── seg.npy
│               ├── dinov3_cls.npy
│               ├── dinov3_patches.npy
│               ├── caption.txt
│               ├── t5_hidden.npy
│               ├── t5_mask.npy
│               └── metadata.json
└── collages/              # Generated visual review aids (optional)
```

---

## File Map (What Goes Where)

| New file | Derives from | Notes |
|----------|-------------|-------|
| `tools/hegre_dataset/__init__.py` | — | Package init, exports public API |
| `tools/hegre_dataset/cli.py` | — | Single entry point: `python -m tools.hegre_dataset <command>` |
| `tools/hegre_dataset/identity.py` | `scripts/09_plan_overnight_enrich.py` | Identity discovery, suffix-aware key, ground-truth scanning |
| `tools/hegre_dataset/face_extraction.py` | `scripts/35_build_hegre_face_dataset.py` | MTCNN detection, multi-face, square crop (shift/clamp, no padding), lineage-preserving output |
| `tools/hegre_dataset/review/__init__.py` | — | Review subpackage |
| `tools/hegre_dataset/review/schema.py` | `scripts/13_seed_review_db.py` | SQLite DDL: personas, sets, images (+ face_index column) |
| `tools/hegre_dataset/review/seed.py` | `scripts/13_seed_review_db.py` + `17_merge_batch.py` | Populate DB from identity discovery + face extraction output |
| `tools/hegre_dataset/review/ui.py` | `scripts/15_review_ui.py` | Flask web UI — shows ACTUAL MTCNN face crops, brush-to-taint |
| `tools/hegre_dataset/review/import_json.py` | `scripts/14_import_review_json.py` | Import pre-existing review decisions |
| `tools/hegre_dataset/enrichment.py` | `scripts/16_batch_resumable_enrich.py` | stratum-hq interface: batch list builder, idempotent enrichment runner |
| `tools/hegre_dataset/collages.py` | `scripts/11_build_120_collages.py` + `12_build_html_gallery.py` | Visual review aids — collages from actual face crops (not keypoint crops) |
| `tools/hegre_dataset/catalog.py` | — | Track dataset versions, adopt existing v0, metadata.json management |
| `tools/hegre_dataset/README.md` | `hegre-dataset-review-system.md` | Rewritten for standalone tool |

---

## Tasks

### Task 1: Create package skeleton and directory structure

**Objective:** Create the `tools/hegre_dataset/` package with empty modules and `__init__.py`.

**Files:**
- Create: `tools/hegre_dataset/__init__.py`
- Create: `tools/hegre_dataset/cli.py` (stub with argparse, no commands wired)
- Create: `tools/hegre_dataset/identity.py` (empty)
- Create: `tools/hegre_dataset/face_extraction.py` (empty)
- Create: `tools/hegre_dataset/review/__init__.py` (empty)
- Create: `tools/hegre_dataset/review/schema.py` (empty)
- Create: `tools/hegre_dataset/review/seed.py` (empty)
- Create: `tools/hegre_dataset/review/ui.py` (empty)
- Create: `tools/hegre_dataset/review/import_json.py` (empty)
- Create: `tools/hegre_dataset/enrichment.py` (empty)
- Create: `tools/hegre_dataset/catalog.py` (empty)
- Create: `tools/hegre_dataset/README.md` (stub)

**Step 1:** Setup NAS storage, symlink, and package directories:
```bash
mkdir -p /mnt/nas-ai-models/training-data/eidolon/hegre_datasets/
mkdir -p data
ln -sfn /mnt/nas-ai-models/training-data/eidolon/hegre_datasets data/hegre_datasets
mkdir -p tools/hegre_dataset/review
```

**Step 2:** Write `tools/hegre_dataset/__init__.py`:
```python
"""Hegre face-dataset creation and management tool.

Commands:
    discover       Scan ground truth for identities and plan extraction.
    extract-faces  Run MTCNN face detection and produce face crops.
    review         Start the Flask review UI.
    enrich         Run stratum-hq enrichment on approved face crops.
    export         Export a gate-ready dataset to a target directory.
    catalog        List and manage dataset versions.
"""
__version__ = "0.1.0"
```

**Step 3:** Write `tools/hegre_dataset/cli.py`:
```python
"""CLI entry point for the hegre dataset tool.

Usage: python -m tools.hegre_dataset <command> [args...]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="hegre-dataset",
        description="Create and manage hegre face datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("discover", help="Discover identities from ground truth")
    sub.add_parser("extract-faces", help="Run MTCNN face detection")
    sub.add_parser("review", help="Start the review UI")
    sub.add_parser("enrich", help="Run stratum-hq enrichment")
    sub.add_parser("export", help="Export gate-ready dataset")
    sub.add_parser("catalog", help="List/manage dataset versions")

    args = parser.parse_args()
    print(f"Command '{args.command}' not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4:** Verify: `python -m tools.hegre_dataset discover` prints "not yet implemented" and exits 1.

**Step 5:** Commit: `git add tools/ && git commit -m "feat: hegre_dataset package skeleton"`

---

### Task 2: Implement identity discovery module

**Objective:** Port `09_plan_overnight_enrich.py` logic to `identity.py` — scan ground truth, discover solo identities, build identity-to-image mapping. Output is a JSON manifest, not a fixed 20-per-identity round-robin sample. All images are included.

**Files:**
- Modify: `tools/hegre_dataset/identity.py`

**Step 1:** Write `tools/hegre_dataset/identity.py`:

Core functions from `scripts/09_plan_overnight_enrich.py`:
- `idkey(slug: str) -> str` — suffix-aware identity key (same algorithm)
- `discover_identities(root: Path, min_sets: int = 3) -> dict[str, list[str]]` — returns `{identity_key: [set_dir_name, ...]}`

New functions:
- `build_manifest(root: Path, identities: dict[str, list[str]]) -> dict` — returns `{identity_key: [{set_slug, image_path, filename}, ...]}` with ALL images from ALL selected sets
- `save_manifest(manifest: dict, output_dir: Path) -> Path` — writes `manifest.json` to dataset directory

```python
"""Identity discovery and image manifest construction.

Scans the hegre ground truth directory, groups images by identity
using suffix-aware keys, and produces a manifest of all images
eligible for face extraction.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def idkey(slug: str) -> str:
    """Extract a suffix-aware identity key from a set slug.
    
    Examples:
        'darina-l' -> 'darina-l'  (suffix '-l' distinguishes from 'darina')
        'keity-climbing' -> 'keity'
        'muriel' -> 'muriel'
    """
    t = slug.split("-")
    k = t[0]
    if len(t) > 1 and len(t[1]) <= 2 and t[1].isalpha():
        k = f"{t[0]}-{t[1]}"
    return k


def discover_identities(root: Path, min_sets: int = 3) -> dict[str, list[str]]:
    """Scan ground truth for identities with at least min_sets photo shoots.
    
    Args:
        root: Path to ground truth directory (e.g., /mnt/.../hegre-14000px/).
        min_sets: Minimum number of distinct shoots an identity must have.
    
    Returns:
        {identity_key: [set_dir_name, ...]} sorted by set count descending.
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    
    for d in sorted(os.listdir(root)):
        if not re.match(r'^\d+_', d):
            continue
        slug = d.split("_", 1)[1]
        by_id[idkey(slug)].append(d)
    
    ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)
    return {k: sets for k, sets in ranked if len(sets) >= min_sets}


def build_manifest(
    root: Path,
    identities: dict[str, list[str]],
    max_identities: int | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Build a manifest of all images for selected identities.
    
    Every image in every set for each identity is included.
    No round-robin sampling — all images are surfaced for review.
    
    Args:
        root: Path to ground truth directory.
        identities: {identity_key: [set_dir_name, ...]} from discover_identities.
        max_identities: If set, limit to top-N identities by set count.
    
    Returns:
        {identity_key: [{set_slug, image_path, filename}, ...]}
    """
    if max_identities:
        identities = dict(list(identities.items())[:max_identities])
    
    manifest: dict[str, list[dict[str, str]]] = {}
    
    for model, set_dirs in identities.items():
        entries: list[dict[str, str]] = []
        for s in set_dirs:
            set_path = root / s
            slug = s.split("_", 1)[1]
            for f in sorted(os.listdir(set_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("_"):
                    entries.append({
                        "set_slug": slug,
                        "filename": f,
                        "image_path": str(set_path / f),
                    })
        manifest[model] = entries
    
    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write manifest.json to the dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
```

**Step 2:** Wire the `discover` subcommand in `cli.py`:

```python
def cmd_discover(args):
    from .identity import discover_identities, build_manifest, save_manifest
    from pathlib import Path
    
    root = Path(args.source)
    identities = discover_identities(root, min_sets=args.min_sets)
    print(f"Found {len(identities)} identities with ≥{args.min_sets} sets.")
    
    manifest = build_manifest(root, identities, max_identities=args.max_identities)
    total_images = sum(len(v) for v in manifest.values())
    print(f"Manifest: {len(manifest)} identities, {total_images} images total.")
    
    output = Path(args.dataset)
    path = save_manifest(manifest, output)
    print(f"Saved: {path}")
    return 0
```

**Step 3:** Add `discover` subcommand arguments:
- `--source` (required): path to ground truth
- `--dataset` (required): path to dataset version directory
- `--min-sets` (default 3): min shoots per identity
- `--max-identities` (optional): cap identity count

**Step 4:** Verify: `python -m tools.hegre_dataset discover --source /tmp/test --dataset /tmp/test_out` prints correct output (even with a small ground truth subset).

**Step 5:** Commit: `git add -A && git commit -m "feat: identity discovery module with manifest builder"`

---

### Task 3: Implement MTCNN face extraction with multi-face detection

**Objective:** Port `35_build_hegre_face_dataset.py` to `face_extraction.py`. Key changes: multi-face detection (extract ALL faces, name `_face1`, `_face2`, ...), idempotency (skip existing), lineage-preserving directory mirroring, shift/clamp square crop (no padding). Same MTCNN config, same `get_square_box()` algorithm.

**Files:**
- Modify: `tools/hegre_dataset/face_extraction.py`

**Step 1:** Write `tools/hegre_dataset/face_extraction.py`:

```python
"""MTCNN face detection and cropping with multi-face support.

Extracts all detectable faces from source images using MTCNN.
Each face is saved with a _face{N} suffix. The square crop uses
shift/clamp (no zero-padding) to avoid polluting ViT embeddings.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None

# MTCNN instance — created once, reused across calls.
_mtcnn: MTCNN | None = None


def get_mtcnn(device: str = "cuda:0") -> MTCNN:
    """Get or create the MTCNN detector (singleton per process)."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=torch.device(device))
    return _mtcnn


def get_square_box(
    box: list[float],
    img_width: int,
    img_height: int,
    expand_ratio: float = 1.5,
) -> list[int]:
    """Compute a square crop box from a face detection, shifted/clamped to image bounds.
    
    No zero-padding — the box is shifted to stay fully inside the source image.
    If the square is larger than the image, clamps to the maximum available size.
    Follows the ViT cropping pattern (plan skill reference).
    
    Args:
        box: [x1, y1, x2, y2] from MTCNN.
        img_width: Source image width.
        img_height: Source image height.
        expand_ratio: Multiplier on the larger box dimension for margin.
    
    Returns:
        [x1, y1, x2, y2] integer coordinates, guaranteed within bounds.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2

    # Expand based on larger dimension
    side = max(w, h) * expand_ratio

    # Attempt to center
    nx1, ny1 = cx - side / 2, cy - side / 2
    nx2, ny2 = cx + side / 2, cy + side / 2

    # Shift if out of bounds (instead of padding)
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_width:
        nx1 -= nx2 - img_width
        nx2 = img_width
    if ny2 > img_height:
        ny1 -= ny2 - img_height
        ny2 = img_height

    # Clamp
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)

    # Enforce square on the clamped size
    final_side = min(nx2 - nx1, ny2 - ny1)
    fx1 = cx - final_side / 2
    fy1 = cy - final_side / 2
    fx2 = cx + final_side / 2
    fy2 = cy + final_side / 2

    return [max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2)]


def extract_faces(
    image_path: str,
    output_dir: Path,
    identity: str,
    set_slug: str,
    filename: str,
    mtcnn: MTCNN | None = None,
    max_dim: int = 1024,
    expand_ratio: float = 1.5,
) -> list[str]:
    """Extract all faces from a single image.
    
    Args:
        image_path: Absolute path to source JPEG.
        output_dir: Dataset root directory.
        identity: Identity key (e.g., 'keity').
        set_slug: Photo shoot slug (e.g., 'keity-climbing').
        filename: Original image filename (e.g., 'keity-climbing-01-3000px.jpg').
        mtcnn: MTCNN instance (created if None).
        max_dim: Maximum crop dimension (downscale if larger).
        expand_ratio: Square box expansion factor.
    
    Returns:
        List of saved relative paths (e.g., ['keity/keity-climbing/keity-climbing-01-3000px_face1.jpg']).
        Empty list if no faces detected.
    """
    if mtcnn is None:
        mtcnn = get_mtcnn()
    
    name_stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # Lineage: mirror source structure
    out_dir = output_dir / "faces" / identity / set_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ERROR opening {image_path}: {e}")
        return []
    
    # MTCNN detect
    try:
        boxes, probs = mtcnn.detect(img)
    except Exception as e:
        print(f"  ERROR detecting {image_path}: {e}")
        return []
    
    if boxes is None or len(boxes) == 0:
        return []
    
    saved = []
    for i, box in enumerate(boxes):
        face_index = i + 1
        out_name = f"{name_stem}_face{face_index}{ext}"
        out_path = out_dir / out_name
        
        # Idempotency
        if out_path.exists():
            saved.append(str(out_path.relative_to(output_dir)))
            continue
        
        try:
            sq_box = get_square_box(box, img.width, img.height, expand_ratio)
            face_crop = img.crop(tuple(sq_box))
            
            # Downscale if larger than max_dim; never upscale
            if face_crop.width > max_dim or face_crop.height > max_dim:
                face_crop = face_crop.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            face_crop.save(out_path, quality=95)
            saved.append(str(out_path.relative_to(output_dir)))
        except Exception as e:
            print(f"  ERROR cropping face {i} from {filename}: {e}")
    
    return saved


def extract_all(
    manifest: dict,
    output_dir: Path,
    device: str = "cuda:0",
    max_workers: int = 4,
    max_dim: int = 1024,
    expand_ratio: float = 1.5,
) -> dict[str, list[str]]:
    """Extract faces for all images in the manifest.
    
    ThreadPoolExecutor for concurrent MTCNN inference (MTCNN is
    thread-safe; GPU serialization is handled internally).
    
    Returns:
        Tuple of ({image_path: [saved_relative_path, ...]}, {identity: zero_detection_count})
    """
    mtcnn = get_mtcnn(device)
    results: dict[str, list[str]] = {}
    
    # Flatten manifest into a list of (identity, set_slug, image_path, filename)
    tasks = []
    for identity, entries in manifest.items():
        for entry in entries:
            tasks.append((
                entry["image_path"],
                identity,
                entry["set_slug"],
                entry["filename"],
            ))
    
    def _process(task):
        img_path, ident, slug, fname = task
        saved = extract_faces(
            img_path, output_dir, ident, slug, fname,
            mtcnn=mtcnn, max_dim=max_dim, expand_ratio=expand_ratio,
        )
        return img_path, saved
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(_process, tasks))
    
    zero_detections = {ident: 0 for ident in manifest.keys()}
    
    for img_path, saved in futures:
        results[img_path] = saved
        if not saved:
            # find identity for this image to increment miss counter
            for ident, entries in manifest.items():
                if any(e["image_path"] == img_path for e in entries):
                    zero_detections[ident] += 1
                    break
    
    n_faces = sum(len(v) for v in results.values())
    n_images = len(results)
    n_with_faces = sum(1 for v in results.values() if v)
    print(f"Extracted {n_faces} faces from {n_with_faces}/{n_images} images.")
    print(f"Zero detections: {n_images - n_with_faces} images skipped.")
    
    # Save zero-detection stats to metadata
    meta_path = output_dir / "zero_detections.json"
    meta_path.write_text(json.dumps(zero_detections, indent=2))
    
    return results, zero_detections
```

**Step 2:** Wire the `extract-faces` subcommand in `cli.py`:
- `--dataset` (required): path to dataset directory (reads `manifest.json`, writes to `faces/`)
- `--device` (default `cuda:0`)
- `--max-workers` (default 4)
- `--max-dim` (default 1024)
- `--expand-ratio` (default 1.5)

**Step 3:** Verify: Run on one known image from the ground truth and inspect the output crop visually.

**Step 4:** Commit: `git add -A && git commit -m "feat: MTCNN face extraction with multi-face support"`

---

### Task 4: Implement review database schema and seeding

**Objective:** Port the SQLite schema and seeding logic. New: `images` table gains a `face_index` column. Seeding reads the face extraction output (not the ground truth directly).

**Files:**
- Modify: `tools/hegre_dataset/review/schema.py`
- Modify: `tools/hegre_dataset/review/seed.py`

**Step 1:** Write `tools/hegre_dataset/review/schema.py`:

```python
"""SQLite schema for the hegre dataset review system."""
import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER NOT NULL,
    slug TEXT NOT NULL,
    FOREIGN KEY (persona_id) REFERENCES personas(id),
    UNIQUE(persona_id, slug)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER NOT NULL,
    set_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,       -- path to the face-crop JPEG (relative to dataset root)
    source_image TEXT NOT NULL,     -- original ground-truth image path (for lineage)
    face_index INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_at TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id),
    FOREIGN KEY (set_id) REFERENCES sets(id),
    UNIQUE(persona_id, set_id, face_index)
);

CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_persona ON images(persona_id);
"""


def get_db(db_path: Path) -> sqlite3.Connection:
    """Open the review database and ensure schema exists."""
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    db.commit()
    return db
```

**Step 2:** Write `tools/hegre_dataset/review/seed.py`:

```python
"""Seed the review database from face extraction output."""
import json
from pathlib import Path

from .schema import get_db


def seed_from_extraction(
    db_path: Path,
    faces_dir: Path,
    manifest_path: Path,
) -> int:
    """Populate the review database from extracted face crops.
    
    Reads the manifest to get identity→set→image mappings.
    Scans faces_dir for extracted _face{N} crops.
    Inserts personas, sets, and images with status='unreviewed'.
    
    Returns:
        Number of images inserted.
    """
    db = get_db(db_path)
    manifest = json.loads(manifest_path.read_text())
    
    total = 0
    for identity, entries in manifest.items():
        # Insert or get persona
        db.execute(
            "INSERT OR IGNORE INTO personas (name) VALUES (?)",
            (identity,),
        )
        persona_id = db.execute(
            "SELECT id FROM personas WHERE name = ?", (identity,)
        ).fetchone()["id"]
        
        # Group entries by set_slug
        sets_seen: set[str] = set()
        
        for entry in entries:
            set_slug = entry["set_slug"]
            source_image = entry["image_path"]
            filename = entry["filename"]
            name_stem = Path(filename).stem
            
            # Insert or get set
            if set_slug not in sets_seen:
                db.execute(
                    "INSERT OR IGNORE INTO sets (persona_id, slug) VALUES (?, ?)",
                    (persona_id, set_slug),
                )
                sets_seen.add(set_slug)
            
            set_id = db.execute(
                "SELECT id FROM sets WHERE persona_id = ? AND slug = ?",
                (persona_id, set_slug),
            ).fetchone()["id"]
            
            # Find all extracted face crops for this source image
            face_dir = faces_dir / identity / set_slug
            if not face_dir.exists():
                continue
            
            face_files = sorted(
                f for f in face_dir.iterdir()
                if f.name.startswith(f"{name_stem}_face") and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            
            for ff in face_files:
                # Extract face_index from filename
                face_name = ff.stem  # e.g., "keity-climbing-01-3000px_face2"
                try:
                    face_index = int(face_name.rsplit("_face", 1)[1])
                except (ValueError, IndexError):
                    face_index = 1
                
                relative_path = str(ff.relative_to(faces_dir.parent))
                
                db.execute(
                    """INSERT OR IGNORE INTO images 
                       (persona_id, set_id, image_path, source_image, face_index, status)
                       VALUES (?, ?, ?, ?, ?, 'unreviewed')""",
                    (persona_id, set_id, relative_path, source_image, face_index),
                )
                total += db.execute("SELECT changes()").fetchone()[0]
    
    db.commit()
    db.close()
    return total
```

**Step 3:** Wire `review seed` subcommand in `cli.py`.

**Step 4:** Verify: Create a small test manifest + face crops in a temporary directory (e.g., `/tmp/hegre_test_dataset`), seed its `review.db`, check row counts with `sqlite3`. DO NOT touch the live `data/review.db`.

**Step 5:** Commit.

---

### Task 5: Implement the review UI (Flask, shows actual MTCNN crops)

**Objective:** Port `15_review_ui.py` to `review/ui.py`. Critical change: the `/api/thumb/<id>` endpoint serves the ACTUAL MTCNN face crop JPEG from `faces/` directory, not a DWPose-keypoint crop computed on-the-fly. The `_crop_face()` function and DWPose keypoint dependency are removed entirely.

**Files:**
- Modify: `tools/hegre_dataset/review/ui.py`

**Step 1:** Write `tools/hegre_dataset/review/ui.py`.

Key differences from the original:
- `image_path` in the DB now points to the face crop JPEG (not the ground truth editorial image).
- `/api/thumb/<id>` loads the crop JPEG directly, resizes to a thumbnail, and serves it.
- No `pose.npy` dependency. No `FACE_SLICE`. No `np.load()`.
- The HTML/CSS/JS is identical in structure (brush buttons, DONE, grid).
- Database schema: adds `face_index` column; the UI groups by persona and shows all face crops.

```python
"""Interactive review UI for hegre face datasets.

Shows actual MTCNN face crops for visual verification.
Brush-to-taint, DONE-to-approve. Port-configurable Flask server.
"""
import io
import sqlite3
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image


def get_db(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db


def create_app(db_path: Path, faces_root: Path) -> Flask:
    """Create the Flask application.
    
    Args:
        db_path: Path to review.db.
        faces_root: Path to the faces/ directory (contains identity/set/ crops).
    """
    app = Flask(__name__)
    
    _thumb_cache: dict[int, bytes] = {}
    THUMB_SIZE = (120, 120)
    
    def _load_thumb(image_path_rel: str) -> bytes:
        """Load and resize a face crop to thumbnail size."""
        full_path = faces_root / image_path_rel
        if not full_path.exists():
            # Return a gray placeholder for missing files
            placeholder = Image.new("RGB", THUMB_SIZE, (60, 60, 60))
            buf = io.BytesIO()
            placeholder.save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        
        img = Image.open(full_path).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
    
    @app.route("/api/thumb/<int:image_id>")
    def api_thumb(image_id):
        if image_id not in _thumb_cache:
            db = get_db(db_path)
            row = db.execute(
                "SELECT image_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if not row:
                return "", 404
            _thumb_cache[image_id] = _load_thumb(row["image_path"])
            if len(_thumb_cache) > 200:
                _thumb_cache.pop(next(iter(_thumb_cache)))
        return send_file(io.BytesIO(_thumb_cache[image_id]), mimetype="image/jpeg")
    
    @app.route("/api/random_persona")
    def api_random_persona():
        mode = request.args.get("mode", "unreviewed")
        status_filter = "approved" if mode == "review" else "unreviewed"
        db = get_db(db_path)
        
        row = db.execute(
            f"""SELECT p.id, p.name FROM personas p
                JOIN images i ON i.persona_id = p.id
                WHERE i.status = ?
                GROUP BY p.id
                ORDER BY RANDOM() LIMIT 1""",
            (status_filter,),
        ).fetchone()
        
        if not row:
            msg = "ALL REVIEWED" if mode == "review" else "ALL DONE"
            return jsonify({"persona_id": None, "persona_name": msg, "image_ids": [], "mode": mode})
        
        pid, pname = row["id"], row["name"]
        all_imgs = db.execute(
            "SELECT id, status, face_index, image_path FROM images WHERE persona_id = ? ORDER BY RANDOM()",
            (pid,),
        ).fetchall()
        
        return jsonify({
            "persona_id": pid,
            "persona_name": pname,
            "image_ids": [r["id"] for r in all_imgs],
            "unreviewed_ids": [r["id"] for r in all_imgs if r["status"] == status_filter],
            "statuses": {r["id"]: r["status"] for r in all_imgs},
            "labels": {r["id"]: f"face{r['face_index']}" for r in all_imgs},
            "mode": mode,
        })
    
    @app.route("/api/done", methods=["POST"])
    def api_done():
        data = request.get_json()
        pid = data["persona_id"]
        tainted = data.get("tainted", {})
        mode = data.get("mode", "unreviewed")
        db = get_db(db_path)
        
        for img_id_str, reason in tainted.items():
            db.execute(
                "UPDATE images SET status = ?, reviewed_at = datetime('now') WHERE id = ?",
                (reason, int(img_id_str)),
            )
        
        if mode != "review":
            db.execute(
                "UPDATE images SET status = 'approved', reviewed_at = datetime('now') WHERE persona_id = ? AND status = 'unreviewed'",
                (pid,),
            )
        
        db.commit()
        
        remaining = db.execute(
            "SELECT COUNT(*) FROM images WHERE status = ?",
            (status_filter,),
        ).fetchone()[0]
        
        return jsonify({"remaining": remaining, "mode": mode})
    
    # HTML template (same structure as original, adapted for face-index labels)
    HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body { background:#1a1a1a; color:#ddd; font-family:sans-serif; margin:20px; }
h2 { color:#4CAF50; }
.grid { display:flex; flex-wrap:wrap; gap:6px; }
.thumb { width:120px; height:120px; object-fit:cover; border:2px solid #333; cursor:pointer; transition:border-color 0.2s,opacity 0.2s; }
.thumb.tainted-black { border-color:#000; opacity:0.5; }
.thumb.tainted-nonface { border-color:#f44336; opacity:0.5; }
.thumb.tainted-contamination { border-color:#e91e63; opacity:0.5; }
.thumb.tainted-unusable { border-color:#9C27B0; opacity:0.5; }
.thumb.approved { border-color:#4CAF50; opacity:0.6; }
.thumb.unreviewed { border-color:#555; }
.tools { margin:12px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.brush { padding:8px 16px; border:none; cursor:pointer; font-weight:bold; border-radius:4px; color:white; }
.brush.nonface { background:#f44336; }
.brush.contamination { background:#e91e63; }
.brush.unusable { background:#9C27B0; }
.brush.done { background:#4CAF50; font-size:16px; padding:10px 24px; }
.brush.mode { background:#555; font-size:12px; }
.brush:hover { opacity:0.85; } .brush.active { outline:3px solid white; }
.key { font-size:12px; color:#666; margin:8px 0; }
.label { font-size:10px; color:#888; text-align:center; margin-top:2px; }
.thumb-wrapper { display:flex; flex-direction:column; align-items:center; }
</style></head><body>
<h2><span id="persona_name">loading...</span></h2>
<div class="key"><span style="color:#555">■ unreviewed</span> <span style="color:#4CAF50">■ approved</span> <span style="color:#f44336">■ non-face</span> <span style="color:#e91e63">■ contamination</span> <span style="color:#9C27B0">■ unusable</span></div>
<div class="tools">
  <span style="color:#aaa">Brush:</span>
  <button class="brush nonface active" id="btn_nonface" onclick="setBrush('tainted:extraction_nonface')">Non-face</button>
  <button class="brush contamination" id="btn_contam" onclick="setBrush('tainted:contamination')">Contamination</button>
  <button class="brush unusable" id="btn_unusable" onclick="setBrush('tainted:unusable')">Unusable</button>
  <button class="brush done" onclick="donePersona()">&#x2713; DONE</button>
  <button class="brush mode" id="btn_review_mode" onclick="switchMode('review')">&#x21BB; Review Pass</button>
  <button class="brush mode" id="btn_unreviewed_mode" style="display:none" onclick="switchMode('unreviewed')">&#x21E0; First Pass</button>
  <span id="status"></span>
</div>
<div class="grid" id="grid"></div>
<script>
let personaId=null,brush='tainted:extraction_nonface',tainted={},mode='unreviewed';
function switchMode(m){mode=m;document.getElementById('btn_review_mode').style.display=m==='review'?'none':'inline-block';document.getElementById('btn_unreviewed_mode').style.display=m==='review'?'inline-block':'none';loadPersona();}
function setBrush(b){brush=b;document.querySelectorAll('.brush').forEach(e=>e.classList.remove('active'));if(b==='tainted:extraction_nonface')document.getElementById('btn_nonface').classList.add('active');if(b==='tainted:contamination')document.getElementById('btn_contam').classList.add('active');if(b==='tainted:unusable')document.getElementById('btn_unusable').classList.add('active');}
async function loadPersona(){
  const resp=await fetch('/api/random_persona?mode='+mode);
  const data=await resp.json();
  if(!data.persona_id){document.getElementById('grid').innerHTML='<p style="font-size:24px;color:#4CAF50">'+data.persona_name+'!</p>';return;}
  personaId=data.persona_id;document.getElementById('persona_name').innerText=data.persona_name;
  const n=data.unreviewed_ids.length;document.getElementById('status').innerText=n+' '+(mode==='review'?'approved':'unreviewed');
  tainted={};
  renderGrid(data.image_ids,data.statuses,data.labels);
}
function renderGrid(ids,statuses,labels){
  const grid=document.getElementById('grid');grid.innerHTML='';
  for(const id of ids){
    const s=statuses[id]||'unreviewed';const lbl=labels[id]||'';
    const wrapper=document.createElement('div');wrapper.className='thumb-wrapper';
    const img=document.createElement('img');img.src='/api/thumb/'+id;img.dataset.id=id;img.className='thumb';
    if(s.startsWith('tainted:')){img.classList.add('tainted-'+s.replace('tainted:extraction_','').replace('tainted:',''));}
    else if(s==='approved'){img.classList.add('approved');if(mode==='review')img.onclick=()=>toggleTaint(img,id);}
    else{img.classList.add('unreviewed');img.onclick=()=>toggleTaint(img,id);}
    wrapper.appendChild(img);
    if(lbl){const label=document.createElement('div');label.className='label';label.innerText=lbl;wrapper.appendChild(label);}
    grid.appendChild(wrapper);
  }
}
function toggleTaint(el,id){
  if(tainted[id]){delete tainted[id];el.className='thumb';if(mode==='review')el.classList.add('approved');else el.classList.add('unreviewed');}
  else{tainted[id]=brush;el.className='thumb';el.classList.add('tainted-'+brush.replace('tainted:extraction_','').replace('tainted:',''));}
}
async function donePersona(){
  const t=Object.keys(tainted).length;
  const msg=mode==='review'?('Re-taint '+t+' images? (rest stay approved)'):('Approve all unreviewed images? ('+t+' tainted, rest approved)');
  if(!confirm(msg))return;
  const resp=await fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({persona_id:personaId,tainted:tainted,mode:mode})});
  const data=await resp.json();
  document.getElementById('status').innerText='Saved. '+data.remaining+' items remaining. Loading next...';
  setTimeout(loadPersona,400);
}
loadPersona();
</script></body></html>"""
    
    @app.route("/")
    def index():
        return render_template_string(HTML)
    
    return app


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Hegre dataset review UI")
    ap.add_argument("--dataset", required=True, help="Path to dataset directory")
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()
    
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    faces_root = dataset.parent  # image_path is relative to dataset root
    
    app = create_app(db_path, faces_root)
    print(f"Review UI at http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
```

**Step 2:** Wire `review` subcommand in `cli.py` to call `review.ui.main()` via `subprocess` or direct import.

**Step 3:** Verify: Start the UI with the temporary test database created in Task 4, confirm thumbnails are served from actual face crops. DO NOT use the live `data/review.db`.

**Step 4:** Commit.

---

### Task 6: Implement stratum enrichment interface

**Objective:** Port `16_batch_resumable_enrich.py` to `enrichment.py`. Builds resumable batch lists from approved face crops, invokes stratum-hq CLI.

**Files:**
- Modify: `tools/hegre_dataset/enrichment.py`

**Step 1:** Write `tools/hegre_dataset/enrichment.py`:

```python
"""stratum-hq enrichment interface for hegre face datasets.

Builds resumable batch image lists from approved face crops
and invokes the stratum-hq CLI for enrichment passes.
"""
import json
import sqlite3
import subprocess
from pathlib import Path


BATCH_SIZE = 250
DEFAULT_PASSES = ["pose", "seg", "depth", "normal", "dinov3", "t5"]


def get_approved_images(db_path: Path, faces_root: Path) -> list[str]:
    """Query review.db for approved face crop paths.
    
    Returns:
        List of absolute paths to approved face crop JPEGs.
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT image_path FROM images WHERE status = 'approved'"
    ).fetchall()
    db.close()
    return [str(faces_root / row[0]) for row in rows]


def build_batches(
    image_paths: list[str],
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> list[Path]:
    """Split image paths into batch text files for stratum-hq.
    
    Returns:
        List of paths to batch_NN.txt files.
    """
    batches = [
        image_paths[i : i + batch_size]
        for i in range(0, len(image_paths), batch_size)
    ]
    
    batch_files = []
    for bi, batch in enumerate(batches):
        out_file = output_dir / f"batch_{bi + 1:02d}.txt"
        out_file.write_text("\n".join(batch) + "\n")
        batch_files.append(out_file)
    
    return batch_files


def run_enrichment(
    batch_file: Path,
    output_root: Path,
    passes: list[str] | None = None,
    device: str = "cuda:0",
) -> subprocess.CompletedProcess:
    """Run stratum-hq on a single batch.
    
    stratum-hq inherently skips already-enriched images,
    so re-running a batch is safe and idempotent.
    """
    if passes is None:
        passes = DEFAULT_PASSES
    
    cmd = [
        "stratum", "process",
        str(batch_file.parent),  # root (ignored when --image-list is used)
        "--output", str(output_root),
        "--image-list", str(batch_file),
        "--passes", ",".join(passes),
        "--device", device,
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=False)


def enrich_dataset(
    dataset_dir: Path,
    passes: list[str] | None = None,
    device: str = "cuda:0",
    batch_size: int = BATCH_SIZE,
) -> None:
    """Full enrichment workflow for a dataset.
    
    1. Query approved images from review.db.
    2. Build batch image lists.
    3. Run stratum-hq on each batch.
    4. Write enrichment config to metadata.json.
    """
    db_path = dataset_dir / "review.db"
    faces_root = dataset_dir.parent  # face paths are relative to dataset parent
    stratum_dir = dataset_dir / "stratum"
    stratum_dir.mkdir(parents=True, exist_ok=True)
    
    images = get_approved_images(db_path, faces_root)
    if not images:
        print("No approved images found. Run review first.")
        return
    
    print(f"Enriching {len(images)} approved face crops...")
    
    batches = build_batches(images, dataset_dir, batch_size)
    print(f"Split into {len(batches)} batches of ~{batch_size}")
    
    for bf in batches:
        print(f"\n--- {bf.name} ---")
        result = run_enrichment(bf, stratum_dir, passes, device)
        if result.returncode != 0:
            print(f"  WARNING: batch {bf.name} exited with code {result.returncode}")
    
    # Update metadata
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["enrichment"] = {
        "pipeline": "stratum-hq",
        "passes": passes or DEFAULT_PASSES,
        "device": device,
        "batch_size": batch_size,
        "approved_images": len(images),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nDone. Metadata updated: {meta_path}")
```

**Step 2:** Wire `enrich` subcommand in `cli.py`.

**Step 3:** Verify: Not easily testable without stratum-hq installed. The code is designed to fail gracefully and report errors.

**Step 4:** Commit.

---

### Task 7: Implement dataset catalog and metadata management

**Objective:** `catalog.py` tracks dataset versions, records provenance in `metadata.json`, and can "adopt" the existing v0 dataset without moving files.

**Files:**
- Modify: `tools/hegre_dataset/catalog.py`

**Step 1:** Write `tools/hegre_dataset/catalog.py`:

```python
"""Dataset version tracking and metadata management."""
import json
from datetime import datetime, timezone
from pathlib import Path


CATALOG_PATH = Path("data/hegre_datasets/catalog.json")


def init_catalog() -> dict:
    """Load or create the catalog file."""
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CATALOG_PATH.exists():
        return json.loads(CATALOG_PATH.read_text())
    return {"datasets": {}}


def register_dataset(
    version: str,
    dataset_dir: Path,
    metadata: dict | None = None,
) -> None:
    """Register a dataset version in the catalog."""
    catalog = init_catalog()
    catalog["datasets"][version] = {
        "path": str(dataset_dir.resolve()),
        "registered": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, default=str))


def adopt_v0() -> None:
    """Adopt the existing Phase 1–4 dataset as v0 without moving files.
    
    Records the current experiment data locations in the catalog.
    Does NOT copy or move any data.
    """
    v0_meta = {
        "version": "v0",
        "created": "2026-06-10",
        "source": "/mnt/nas-ai-models/training-data/loras/hegre-14000px",
        "notes": (
            "Original dataset built during Eidolon Phases 1–4. "
            "120 identities, 1589 approved face crops. "
            "Face crops in hegre_faces/, enrichment in hegre_faces_stratum/. "
            "Review database at data/review.db. "
            "Frozen — not managed by tools/hegre_dataset."
        ),
        "review_db": "experiments/geometry_pca/data/review.db",
        "faces_tree": "experiments/geometry_pca/data/hegre_faces",
        "stratum_tree": "experiments/geometry_pca/data/hegre_faces_stratum",
        "parent_version": None,
    }
    register_dataset("v0", Path("experiments/geometry_pca/data"), v0_meta)
    print("Adopted existing v0 dataset. Catalog updated.")
```

**Step 2:** Wire `catalog adopt-v0` and `catalog list` subcommands.

**Step 3:** Commit.

---

### Task 8: Annotate superseded experiment scripts

**Objective:** Add a header comment to each superseded script in `experiments/geometry_pca/scripts/` noting it is frozen and superseded by `tools/hegre_dataset/`. No code changes — comment only.

**Files** to annotate (add as line 2, after shebang if present):
- `experiments/geometry_pca/scripts/09_plan_overnight_enrich.py`
- `experiments/geometry_pca/scripts/10_verify_gate.py`
- `experiments/geometry_pca/scripts/11_build_120_collages.py`
- `experiments/geometry_pca/scripts/12_build_html_gallery.py`
- `experiments/geometry_pca/scripts/13_seed_review_db.py`
- `experiments/geometry_pca/scripts/14_import_review_json.py`
- `experiments/geometry_pca/scripts/15_review_ui.py`
- `experiments/geometry_pca/scripts/16_batch_resumable_enrich.py`
- `experiments/geometry_pca/scripts/17_merge_batch.py`
- `experiments/geometry_pca/scripts/35_build_hegre_face_dataset.py`

Comment text:
```
# Superseded by tools/hegre_dataset/ as of 2026-06-12.
# This script is frozen as part of the Phase 1–4 experimental record.
# For new dataset creation, use: python -m tools.hegre_dataset <command>
```

Also annotate `experiments/geometry_pca/hegre-dataset-review-system.md` with a similar note at the top.

**Step 1:** Verify each file exists before patching.

**Step 2:** Apply the comment header insertions.

**Step 3:** Commit: `git commit -m "docs: annotate superseded hegre-dataset scripts (frozen for experimental record)"`

---

### Task 9: Write `README.md` for the tool

**Objective:** Write comprehensive documentation for the standalone tool. Replace the content of `tools/hegre_dataset/README.md`.

**File:** `tools/hegre_dataset/README.md`

Content should cover:
- Purpose and architecture
- Prerequisites (MTCNN via facenet_pytorch, Flask, stratum-hq CLI, CUDA GPU)
- CLI reference for each command
- Dataset directory structure
- Review workflow (brush-to-taint)
- Enrichment workflow (resumable batches)
- How the experiment consumes the output
- Relationship to v0 (the frozen Phase 1–4 dataset)

**Step 1:** Write the README.

**Step 2:** Commit: `git commit -m "docs: hegre_dataset tool README"`

---

### Task 10: Final integration test and validation

**Objective:** Run the full pipeline end-to-end with a tiny subset (1 identity, 1 shoot) to verify all components wire together.

**Step 1:** Create a test dataset:
```bash
python -m tools.hegre_dataset discover \
    --source /mnt/nas-ai-models/training-data/loras/hegre-14000px \
    --dataset data/hegre_datasets/test-v1 \
    --min-sets 1 --max-identities 1
```

**Step 2:** Extract faces:
```bash
python -m tools.hegre_dataset extract-faces \
    --dataset data/hegre_datasets/test-v1 \
    --device cuda:0
```

**Step 3:** Seed the review database:
```bash
python -m tools.hegre_dataset review seed \
    --dataset data/hegre_datasets/test-v1
```

**Step 4:** Verify the database has correct counts:
```bash
sqlite3 data/hegre_datasets/test-v1/review.db \
    "SELECT status, COUNT(*) FROM images GROUP BY status"
```

**Step 5:** Spot-check one face crop visually — confirm it's a proper square crop of a face.

**Step 6:** Clean up test dataset.

**Step 7:** Commit any fixes discovered during integration testing.

---

## Tests / Validation

- **Unit test:** `tests/test_identity.py` — verify `idkey()` suffix-awareness on known examples.
- **Unit test:** `tests/test_face_extraction.py` — verify `get_square_box()` returns in-bounds coordinates for edge cases (face at image border, face larger than image).
- **Unit test:** `tests/test_review_schema.py` — verify DDL creates tables, inserts work, foreign keys enforce.
- **Integration:** Run `discover` → `extract-faces` → `review seed` on a 1-identity subset, verify row counts and crop existence.

## Risks and Tradeoffs

1. **MTCNN dependency:** `facenet_pytorch` must be installed. If unavailable, face extraction fails early with a clear ImportError.
2. **stratum-hq CLI dependency:** The `enrich` command calls `stratum` as a subprocess. If stratum-hq is not on PATH, the command fails with a clear error.
3. **GPU requirement:** MTCNN and stratum-hq both require CUDA. The tool detects CUDA availability and errors early if missing.
4. **Multi-face volume:** 72k source images × ~1.1 faces average = ~80k crops. Review at 120 identities, ~670 crops each. This is a time investment but the tool is designed for incremental review (one persona at a time, resumable).
5. **Existing data safety:** The tool never touches `experiments/geometry_pca/data/`. The v0 adoption is purely a catalog entry (no file operations).
6. **`get_square_box()` preserves the existing algorithm** from `35_build_hegre_face_dataset.py` — no change to cropping behavior. The only behavioral change is multi-face (`keep_all=True` instead of `keep_all=False`).

## Open Questions

None — all design decisions were resolved during the planning conversation.
