#!/usr/bin/env python3
"""
Phase 2: Resumable hegre enrichment for additional identities.
Builds batch image lists so enrichment can be interrupted and resumed.
Each batch is ~250 images (~10 min GPU). Stratum-hq skips already-done images.
"""
import os, re, json
from collections import defaultdict

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
BATCH_SIZE = 250
IMAGES_PER_ID = 20
OUT_PREFIX = "data/batch_"

def idkey(slug):
    t = slug.split('-'); k = t[0]
    if len(t) > 1 and len(t[1]) <= 2 and t[1].isalpha(): k = f"{t[0]}-{t[1]}"
    return k

def couple(slug):
    return ("-and-" in slug) or slug.endswith("-and") or "couple" in slug

def main():
    os.makedirs("data", exist_ok=True)
    
    # Load existing identities from overnight map
    existing = set(json.load(open("data/overnight_identity_map.json")).values())
    
    # Group all solo identities
    by_id = defaultdict(list)
    for d in sorted(os.listdir(ROOT)):
        if not re.match(r'^\d+_', d): continue
        slug = d.split('_', 1)[1]
        if couple(slug): continue
        by_id[idkey(slug)].append(d)
    
    ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    # Pick new identities not already enriched
    new = []
    for k, ss in ranked:
        if k not in existing:
            new.append((k, ss))
    
    # Collect image paths across new identities, round-robin within each
    image_paths = []
    id_map = {}
    for model, ss in new:
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
                    if len(picked) >= IMAGES_PER_ID: break
            idx += 1
        for p in picked:
            image_paths.append(p)
            id_map[os.path.relpath(p, ROOT)] = model
    
    # Split into batches
    batches = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    
    for bi, batch in enumerate(batches):
        out_file = f"{OUT_PREFIX}{bi+1:02d}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join(batch) + "\n")
        out_map = f"{OUT_PREFIX}{bi+1:02d}_map.json"
        batch_map = {os.path.relpath(p, ROOT): id_map[os.path.relpath(p, ROOT)] for p in batch}
        with open(out_map, "w") as f:
            json.dump(batch_map, f, indent=2)
    
    print(f"Total: {len(image_paths)} images across {len(new)} identities")
    print(f"Batches: {len(batches)} x ~{BATCH_SIZE} images each")
    for bi in range(len(batches)):
        print(f"  batch_{bi+1:02d}.txt — {len(batches[bi])} images")
    print(f"\nRun each batch: stratum process <root> --output data/hegre_enriched --image-list data/batch_NN.txt --passes pose,seg,depth,normal --device cuda")
    print(f"When all done, merge batch maps into overnight_identity_map.json")

if __name__ == "__main__":
    main()
