#!/usr/bin/env python3
"""
Count approved Hegre images with zg_distance < 25 that have full text-conditioning data
(caption.txt AND t5_hidden.npy), without extrapolation.

Outputs a count per-persona and a total. Run in tmux — scans all approved zg<25 images
by checking file existence on the NAS via the stratum/faces/ tree.

Usage:
    python scripts/count_full_data_images.py

Expect: ~5-15 minutes for 106k images (NAS exists() calls at ~2ms each = ~200s).
"""
import sqlite3
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

DB_PATH = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
STRATUM_FACES = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum/faces"

def main():
    t0 = time.time()
    
    # Connect read-only
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro&nolock=1", uri=True)
    db.row_factory = sqlite3.Row
    
    rows = db.execute("""
        SELECT p.name as persona, i.image_path
        FROM images i
        JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved' 
          AND i.zg_distance IS NOT NULL 
          AND i.zg_distance < 25.0
        ORDER BY p.name, i.image_path
    """).fetchall()
    db.close()
    
    total = len(rows)
    print(f"Total approved images with zg < 25: {total}")
    print(f"Checking files in {STRATUM_FACES}...")
    print(f"Expected runtime: ~{total * 0.002:.0f}s at ~2ms per exists() call")
    print()
    
    persona_counts = defaultdict(lambda: {"total": 0, "both": 0, "neither": 0})
    have_both = 0
    have_t5 = 0
    have_cap = 0
    checked = 0
    last_report = t0
    
    for row in rows:
        persona = row['persona']
        img_path = row['image_path']  # faces/persona/shoot/img.jpg
        
        # Path construction:
        # DB:   faces/persona/shoot/img.jpg
        # T5:   stratum/faces/persona/shoot/img/t5_hidden.npy
        # Keep faces/ prefix, replace .jpg with /t5_hidden.npy
        stem_path = img_path.replace('.jpg', '/t5_hidden.npy')
        t5_file = STRATUM_FACES + '/' + '/'.join(stem_path.split('/')[1:])
        cap_file = t5_file.replace('/t5_hidden.npy', '/caption.txt')
        
        has_t5 = os.path.exists(t5_file)
        has_cap = os.path.exists(cap_file)
        
        persona_counts[persona]["total"] += 1
        if has_t5:
            have_t5 += 1
        if has_cap:
            have_cap += 1
        if has_t5 and has_cap:
            have_both += 1
            persona_counts[persona]["both"] += 1
        if not has_t5 and not has_cap:
            persona_counts[persona]["neither"] += 1
        
        checked += 1
        
        # Progress report every 5s or 5000 images
        now = time.time()
        if checked % 5000 == 0 or (now - last_report > 5):
            elapsed = now - t0
            rate = checked / elapsed if elapsed > 0 else 0
            eta = (total - checked) / rate if rate > 0 else 0
            pct = checked / total * 100
            print(f"  [{checked}/{total} {pct:.1f}%] "
                  f"both={have_both} t5={have_t5} cap={have_cap} "
                  f"| {rate:.0f} img/s | ETA: {eta:.0f}s",
                  flush=True)
            last_report = now
    
    elapsed = time.time() - t0
    
    # Summary
    print()
    print("=" * 70)
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Total approved, zg < 25: {total}")
    print(f"  Have t5_hidden.npy:  {have_t5}  ({have_t5/total*100:.1f}%)")
    print(f"  Have caption.txt:    {have_cap}  ({have_cap/total*100:.1f}%)")
    print(f"  Have BOTH:           {have_both}  ({have_both/total*100:.1f}%)")
    print()
    
    # Per-persona breakdown
    sorted_personas = sorted(persona_counts.items(), key=lambda x: x[1]["both"], reverse=True)
    print(f"Per-persona breakdown ({len(sorted_personas)} personas):")
    print(f"{'Persona':<32} {'Total':>7} {'Both':>7} {'Neither':>7} {'%Both':>7}")
    print("-" * 65)
    for persona, counts in sorted_personas:
        tot = counts["total"]
        both = counts["both"]
        pct = both / tot * 100 if tot > 0 else 0
        print(f"{persona:<32} {tot:>7} {both:>7} {counts['neither']:>7} {pct:>6.1f}%")
    
    print()
    print(f"RESULT: {have_both} approved images with zg < 25 have full data (caption.txt + t5_hidden.npy)")

if __name__ == "__main__":
    main()
