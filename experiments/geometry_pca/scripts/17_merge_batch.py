#!/usr/bin/env python3
"""Merge a batch map into the review database without affecting existing data."""
import json, sqlite3, sys, os

DB = "data/review.db"

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/17_merge_batch.py data/batch_01_map.json [batch_02_map.json ...]")
        return
    
    db = sqlite3.connect(DB)
    
    for map_path in sys.argv[1:]:
        batch = json.load(open(map_path))
        added = 0
        for rel_path, persona_name in batch.items():
            # Check if persona exists
            pid = db.execute("SELECT id FROM personas WHERE name=?", (persona_name,)).fetchone()
            if not pid:
                db.execute("INSERT INTO personas (name) VALUES (?)", (persona_name,))
                pid = db.execute("SELECT id FROM personas WHERE name=?", (persona_name,)).fetchone()[0]
            else:
                pid = pid[0]
            
            # Set — use the directory name as slug
            set_dir = rel_path.split('/')[0]
            set_slug = set_dir.split('_', 1)[1] if '_' in set_dir else set_dir
            sid = db.execute("SELECT id FROM sets WHERE persona_id=? AND slug=?", (pid, set_slug)).fetchone()
            if not sid:
                db.execute("INSERT INTO sets (persona_id, slug) VALUES (?, ?)", (pid, set_slug))
                sid = db.execute("SELECT id FROM sets WHERE persona_id=? AND slug=?", (pid, set_slug)).fetchone()[0]
            else:
                sid = sid[0]
            
            # Image — only add if not already present
            ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
            image_path = os.path.join(ROOT, rel_path)
            enriched_dir = os.path.join("data/hegre_enriched", rel_path.replace('.jpg', ''))
            
            existing = db.execute("SELECT id FROM images WHERE image_path=?", (image_path,)).fetchone()
            if not existing:
                db.execute(
                    "INSERT INTO images (persona_id, set_id, image_path, enriched_dir, status) VALUES (?, ?, ?, ?, 'unreviewed')",
                    (pid, sid, image_path, enriched_dir))
                added += 1
        
        db.commit()
        print(f"{os.path.basename(map_path)}: {added} new images added")
    
    # Stats
    n_p = db.execute("SELECT COUNT(*) FROM personas").fetchone()[0]
    n_i = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    n_unreviewed = db.execute("SELECT COUNT(*) FROM images WHERE status='unreviewed'").fetchone()[0]
    print(f"DB now: {n_p} personas, {n_i} images, {n_unreviewed} unreviewed")
    db.close()

if __name__ == "__main__":
    main()
