"""Cross-reference pre-refit (low-norm) LDA files against the review DB status.
Read-only."""
import sys, json
from pathlib import Path
import numpy as np
from collections import Counter

ROOT = Path("/home/tim/source/activity/eidolon")
DATASET = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
sys.path.insert(0, str(ROOT))
from tools.hegre_dataset.dataset import HegreDataset

ds = HegreDataset(DATASET)
lda = DATASET / "lda"

# DB map: image_path (relative, e.g. faces/...) -> status
rows = ds.db.execute("SELECT image_path, status FROM images").fetchall()
status_of = {r["image_path"]: r["status"] for r in rows}
print(f"DB rows: {len(status_of)}", flush=True)

low = []
for f in lda.rglob("*.npy"):
    try:
        nrm = float(np.linalg.norm(np.load(f)))
    except Exception:
        continue
    if nrm < 50.0:
        rel = str(f.relative_to(lda))           # faces/.../x.npy
        img = rel[:-4] + ".jpg"                  # corresponding image_path
        low.append((rel, nrm, status_of.get(img, "<NOT IN DB>")))

print(f"low-norm (pre-refit) files: {len(low)}", flush=True)
c = Counter(s for _, _, s in low)
print("status breakdown:", dict(c), flush=True)
approved = [x for x in low if x[2] == "approved"]
print(f"APPROVED with pre-refit LDA: {len(approved)}", flush=True)
for rel, nrm, _ in approved[:40]:
    print(f"   {nrm:8.4f}  {rel}", flush=True)
out = Path("/home/tim/.hermes/profiles/eidolon/cache/scratch/prerefit_lda_files.json")
out.write_text(json.dumps(low, indent=1))
print(f"wrote {out}", flush=True)
print("DONE", flush=True)
