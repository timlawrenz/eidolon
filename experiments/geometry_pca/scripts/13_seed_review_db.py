#!/usr/bin/env python3
"""
Seed the review database from overnight enrichment data.
Creates tables: personas, sets, images.
Populates from overnight_identity_map.json.
"""
import os, sys, json, sqlite3, re

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
ENRICHED = "data/hegre_enriched"
MAP = "data/overnight_identity_map.json"
DB = "data/review.db"

def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    
    db.executescript("""
        CREATE TABLE personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id INTEGER NOT NULL REFERENCES personas(id),
            slug TEXT NOT NULL,
            UNIQUE(persona_id, slug)
        );
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id INTEGER NOT NULL REFERENCES personas(id),
            set_id INTEGER NOT NULL REFERENCES sets(id),
            image_path TEXT NOT NULL UNIQUE,
            enriched_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed',
            reviewed_at TEXT
        );
        CREATE INDEX idx_images_status ON images(status);
        CREATE INDEX idx_images_persona ON images(persona_id);
    """)
    
    mapping = json.load(open(MAP))
    
    persona_cache = {}
    set_cache = {}
    count = 0
    
    for rel_path, persona_name in mapping.items():
        # Persona
        if persona_name not in persona_cache:
            db.execute("INSERT OR IGNORE INTO personas (name) VALUES (?)", (persona_name,))
            pid = db.execute("SELECT id FROM personas WHERE name=?", (persona_name,)).fetchone()[0]
            persona_cache[persona_name] = pid
        else:
            pid = persona_cache[persona_name]
        
        # Set (from the directory name)
        set_dir = rel_path.split('/')[0]
        set_slug = set_dir.split('_', 1)[1] if '_' in set_dir else set_dir
        
        set_key = (pid, set_slug)
        if set_key not in set_cache:
            db.execute("INSERT OR IGNORE INTO sets (persona_id, slug) VALUES (?, ?)", (pid, set_slug))
            sid = db.execute("SELECT id FROM sets WHERE persona_id=? AND slug=?", (pid, set_slug)).fetchone()[0]
            set_cache[set_key] = sid
        else:
            sid = set_cache[set_key]
        
        # Image
        enriched_dir = os.path.join(ENRICHED, rel_path.replace('.jpg', ''))
        image_path = os.path.join(ROOT, rel_path)
        
        db.execute(
            "INSERT OR IGNORE INTO images (persona_id, set_id, image_path, enriched_dir) VALUES (?, ?, ?, ?)",
            (pid, sid, image_path, enriched_dir)
        )
        count += 1
    
    db.commit()
    
    # Stats
    n_p = db.execute("SELECT COUNT(*) FROM personas").fetchone()[0]
    n_s = db.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
    n_i = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"Seeded: {n_p} personas, {n_s} sets, {n_i} images")
    
    # Apply known contaminants from Phase 1-R
    known_bad = ["natalia-a", "muriel"]
    for name in known_bad:
        db.execute("""
            UPDATE images SET status = 'tainted:contamination'
            WHERE persona_id = (SELECT id FROM personas WHERE name = ?)
        """, (name,))
    print(f"Applied Phase 1-R contamination: {known_bad}")
    
    db.close()

if __name__ == "__main__":
    main()
