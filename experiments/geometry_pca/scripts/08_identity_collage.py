#!/usr/bin/env python3
"""Build per-identity face-crop collages from the exact images used in the gate,
so we can visually verify each labeled identity is actually ONE person."""
import os, sys, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/tim/source/activity/prx-tg/scripts")
from dwpose_onnx import DWPoseDetector
from geometry_pca.constants import FACE_SLICE

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
MAX_DIM = 1024
OUT = "output"


def load_rgb(path, max_dim=MAX_DIM):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, max_dim / max(w, h))
    if s < 1.0:
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return np.asarray(im)


def main():
    meta = json.load(open("data/hegre_gate_meta.json"))
    det = DWPoseDetector(device="cpu")

    for model, info in meta.items():
        crops = []
        for rel in info["paths"]:
            path = os.path.join(ROOT, rel)
            try:
                img = load_rgb(path)
                kpts, scores, bboxes = det(img, single_person=True)
                if len(kpts) == 0:
                    continue
                face = kpts[0][FACE_SLICE]
                mn = face.min(axis=0); mx = face.max(axis=0)
                cx, cy = (mn+mx)/2
                half = (mx-mn).max() * 0.8
                x0, y0 = int(max(0, cx-half)), int(max(0, cy-half))
                x1, y1 = int(min(img.shape[1], cx+half)), int(min(img.shape[0], cy+half))
                crop = Image.fromarray(img[y0:y1, x0:x1]).resize((160, 160))
                crops.append(crop)
            except Exception as e:
                print(f"  {model} skip {os.path.basename(rel)}: {e}")
        if not crops:
            continue
        # tile in a grid, 6 per row
        cols = 6
        rows = (len(crops)+cols-1)//cols
        sheet = Image.new("RGB", (cols*160, rows*160), (20,20,20))
        for i, c in enumerate(crops):
            sheet.paste(c, ((i % cols)*160, (i//cols)*160))
        out = os.path.join(OUT, f"collage_{model}.png")
        sheet.save(out)
        print(f"{model}: {len(crops)} faces -> {out}")


if __name__ == "__main__":
    main()
