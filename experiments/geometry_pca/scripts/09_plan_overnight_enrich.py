#!/usr/bin/env python3
"""
Overnight enrichment planner: build an image-list of hegre faces across many
solo identities for stratum-hq enrichment (depth/normal/seg/pose).

Picks top-N solo identities (suffix-aware key, couple sets excluded),
round-robins up to K images per identity across DISTINCT sets, and writes:
  - data/overnight_imagelist.txt   (one absolute image path per line)
  - data/overnight_identity_map.json  (image rel-path -> identity key)

Resumable: stratum itself skips images whose output dir already has the artifacts.
"""
import os, re, json
from collections import defaultdict

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
N_IDENTITIES = 120
IMAGES_PER_ID = 20
OUT_LIST = "data/overnight_imagelist.txt"
OUT_MAP = "data/overnight_identity_map.json"


def idkey(slug):
    t = slug.split('-'); k = t[0]
    if len(t) > 1 and len(t[1]) <= 2 and t[1].isalpha():
        k = f"{t[0]}-{t[1]}"
    return k


def couple(slug):
    return ("-and-" in slug) or slug.endswith("-and") or "couple" in slug


def main():
    os.makedirs("data", exist_ok=True)
    by_id = defaultdict(list)
    for d in sorted(os.listdir(ROOT)):
        if not re.match(r'^\d+_', d):
            continue
        slug = d.split('_', 1)[1]
        if couple(slug):
            continue
        by_id[idkey(slug)].append(d)

    ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)[:N_IDENTITIES]

    image_paths = []
    id_map = {}
    for model, ss in ranked:
        # round-robin one image per distinct set
        per_set = []
        for s in ss:
            imgs = sorted(f for f in os.listdir(os.path.join(ROOT, s)) if f.lower().endswith(".jpg"))
            per_set.append([os.path.join(ROOT, s, f) for f in imgs])
        picked = []
        idx = 0
        while len(picked) < IMAGES_PER_ID and any(idx < len(p) for p in per_set):
            for p in per_set:
                if idx < len(p):
                    picked.append(p[idx])
                    if len(picked) >= IMAGES_PER_ID:
                        break
            idx += 1
        for p in picked:
            image_paths.append(p)
            id_map[os.path.relpath(p, ROOT)] = model

    with open(OUT_LIST, "w") as f:
        f.write("\n".join(image_paths) + "\n")
    with open(OUT_MAP, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"Planned {len(image_paths)} images across {len(ranked)} identities.")
    print(f"  -> {OUT_LIST}\n  -> {OUT_MAP}")


if __name__ == "__main__":
    main()
