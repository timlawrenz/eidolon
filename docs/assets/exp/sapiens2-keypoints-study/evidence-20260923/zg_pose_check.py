"""Are the 9 z_g-less images all-zero DWPose failures (correctly skipped) or a real gap?"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
targets = [
    "faces/dasha-t/dasha-t-rebel-model/dasha-t-rebel-model-board_face1",
    "faces/dasha-t/dasha-t-sculpted/dasha-t-sculpted-24-14000px_face1",
    "faces/elisabeth/elisabeth-forest-nymph/elisabeth-forest-nymph-11-1200px_face1",
    "faces/kasha/kasha-dark-mistress/kasha-dark-mistress-24-1200px_face1",
    "faces/kasha/kasha-oiled/kasha-oiled-26-1200px_face1",
    "faces/kasha/kasha-oiled/kasha-oiled-54-1200px_face1",
    "faces/kasha/kasha-oiled/kasha-oiled-poster_face1",
    "faces/kasha/kasha-tan-addiction/kasha-tan-addiction-51-1200px_face1",
    "faces/olivia/olivia-bare-ballet/olivia-bare-ballet-18-10000px_face1",
]
for t in targets:
    stem = Path(t).stem
    persona = Path(t).parts[1]
    hits = list((ROOT / "stratum" / "faces" / persona).rglob(f"{stem}/pose.npy"))
    if not hits:
        print(f"{stem}: NO pose.npy")
        continue
    p = np.load(hits[0])
    print(f"{Path(t).parent.name}/{stem}: shape={p.shape} dtype={p.dtype} all_zero={bool(np.all(p == 0))} max_abs={float(np.max(np.abs(p))):.4f}")
