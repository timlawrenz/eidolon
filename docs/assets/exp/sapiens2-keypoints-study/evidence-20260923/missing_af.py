"""Characterize the 23 approved images missing AuraFace data."""
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.hegre_dataset.dataset import HegreDataset  # noqa: E402

ds = HegreDataset(ROOT)
rows = ds.db.execute("SELECT image_path FROM images WHERE status = 'approved'").fetchall()
approved = [str(Path(r["image_path"]).with_suffix(".npy")) for r in rows]

af = set()
for dp, _dn, fns in os.walk(ROOT / "auraface"):
    for fn in fns:
        if fn.endswith(".npy"):
            af.add(str((Path(dp) / fn).relative_to(ROOT / "auraface")))

missing = sorted(set(approved) - af)
print(f"total missing: {len(missing)}")
personas = Counter(Path(m).parts[1] for m in missing)
print("by persona:", dict(personas))
for m in missing:
    jpg = ROOT / "faces" / Path(m).relative_to("faces").with_suffix(".jpg")
    print(f"  {m}  crop_exists={jpg.exists()}")
