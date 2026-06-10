#!/usr/bin/env python3
"""
Phase 1-R Step 2a: extract 68 face keypoints from real multi-pose hegre images
for the identity-separability gate. Caches normalized (68,2) arrays to disk so
the z_scale sweep can run instantly in memory.

Identities are drawn across MULTIPLE distinct sets per model so the within-identity
variation reflects real pose/lighting/expression spread (different shoots).
"""
import os
import re
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DWPose detector lives in the prx-tg repo
sys.path.insert(0, "/home/tim/source/activity/prx-tg/scripts")
from dwpose_onnx import DWPoseDetector  # noqa: E402

from geometry_pca.constants import FACE_SLICE  # noqa: E402

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
PICKS = ["alya", "valerie", "yanna", "hiromi", "jessa",
         "anna-l", "darina-l", "dominika-c", "natalia-a", "francy", "inga", "muriel"]
IMAGES_PER_ID = 15          # spread across distinct sets
MAX_DIM = 1024              # downscale long edge for speed
CONF_THRESH = 0.45
OUT = "data/hegre_gate_keypoints.npz"


def _identity_key(slug):
    """Suffix-aware: darina-l is a DIFFERENT person from darina."""
    toks = slug.split('-')
    key = toks[0]
    if len(toks) > 1 and len(toks[1]) <= 2 and toks[1].isalpha():
        key = f"{toks[0]}-{toks[1]}"
    return key


def _is_couple(slug):
    return ("-and-" in slug) or slug.endswith("-and") or "couple" in slug


def sets_for(model):
    out = []
    for d in sorted(os.listdir(ROOT)):
        if not re.match(r'^\d+_', d):
            continue
        slug = d.split('_', 1)[1]
        if _is_couple(slug):
            continue
        if _identity_key(slug) == model:
            out.append(d)
    return out


def list_images_across_sets(model, n_want):
    """Round-robin one image from each distinct set until n_want collected."""
    ss = sets_for(model)
    per_set = []
    for s in ss:
        imgs = sorted(f for f in os.listdir(os.path.join(ROOT, s)) if f.lower().endswith(".jpg"))
        per_set.append([os.path.join(ROOT, s, f) for f in imgs])
    picked = []
    idx = 0
    while len(picked) < n_want and any(idx < len(p) for p in per_set):
        for p in per_set:
            if idx < len(p):
                picked.append(p[idx])
                if len(picked) >= n_want:
                    break
        idx += 1
    return picked


def load_rgb_downscaled(path, max_dim=MAX_DIM):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(im)


def normalize_kpts(kpts68, img_shape):
    """Pixel coords -> [-1,1] centered on the face bbox (match pose.npy convention)."""
    h, w = img_shape[:2]
    xy = kpts68.astype(np.float32)
    # normalize to face bounding box, mapped into [-1,1]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = (mx - mn).max()
    if span < 1e-6:
        return None
    centered = (xy - (mn + mx) / 2.0) / (span / 2.0)
    return centered.astype(np.float32)


def main():
    os.makedirs("data", exist_ok=True)
    print("Loading DWPose detector (CPU)...")
    det = DWPoseDetector(device="cpu")

    data = {}  # identity -> list of (68,2)
    meta = {}
    for model in PICKS:
        paths = list_images_across_sets(model, IMAGES_PER_ID)
        kept = []
        used_paths = []
        for p in paths:
            try:
                img = load_rgb_downscaled(p)
                kpts, scores, bboxes = det(img, single_person=True)
                if len(kpts) == 0:
                    continue
                face = kpts[0][FACE_SLICE]          # (68,2)
                face_score = scores[0][FACE_SLICE]  # (68,)
                if np.mean(face_score) < CONF_THRESH:
                    continue
                norm = normalize_kpts(face, img.shape)
                if norm is None:
                    continue
                kept.append(norm)
                used_paths.append(os.path.relpath(p, ROOT))
            except Exception as e:
                print(f"  skip {os.path.basename(p)}: {e}")
        data[model] = np.stack(kept) if kept else np.zeros((0, 68, 2), np.float32)
        meta[model] = {"n_kept": len(kept), "paths": used_paths}
        print(f"{model:10} kept {len(kept)}/{len(paths)} images")

    # Save: one array per identity + a flat index
    save = {}
    labels = []
    flat = []
    for i, model in enumerate(PICKS):
        save[f"id_{model}"] = data[model]
        for row in data[model]:
            flat.append(row)
            labels.append(i)
    save["X"] = np.stack(flat) if flat else np.zeros((0, 68, 2), np.float32)
    save["y"] = np.array(labels, dtype=np.int32)
    save["names"] = np.array(PICKS)
    np.savez_compressed(OUT, **save)

    with open("data/hegre_gate_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {len(flat)} keypoint sets across {len(PICKS)} identities -> {OUT}")


if __name__ == "__main__":
    main()
