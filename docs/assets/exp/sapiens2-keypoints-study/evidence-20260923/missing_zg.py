"""Confirm/characterize the 9 approved images reported 'with pose but missing z_g'."""
import os
import sys
from pathlib import Path

ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.hegre_dataset.dataset import HegreDataset  # noqa: E402

ds = HegreDataset(ROOT)
rows = ds.db.execute("SELECT image_path FROM images WHERE status = 'approved'").fetchall()
approved = [str(Path(r["image_path"]).with_suffix(".npy")) for r in rows]

zg = set()
for dp, _dn, fns in os.walk(ROOT / "zg"):
    for fn in fns:
        if fn.endswith(".npy"):
            zg.add(str((Path(dp) / fn).relative_to(ROOT / "zg")))

missing = sorted(set(approved) - zg)
print(f"approved images missing z_g: {len(missing)}")
for m in missing:
    stem = Path(m).stem
    persona = Path(m).parts[1]
    stratum_persona = ROOT / "stratum" / "faces" / persona
    poses = list(stratum_persona.rglob(f"{stem}/pose.npy")) if stratum_persona.exists() else []
    print(f"  {m}  pose_exists={bool(poses)}")
