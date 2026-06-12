import json
from pathlib import Path
from .schema import get_db

def seed_from_extraction(db_path: Path, faces_dir: Path, manifest_path: Path, verbose: bool = False) -> int:
    db = get_db(db_path)
    manifest = json.loads(manifest_path.read_text())
    total = 0
    
    if verbose:
        print(f"Loaded manifest with {len(manifest)} identities.")
        print(f"Looking for faces in: {faces_dir}")
    
    for identity, entries in manifest.items():
        db.execute("INSERT OR IGNORE INTO personas (name) VALUES (?)", (identity,))
        persona_id = db.execute("SELECT id FROM personas WHERE name = ?", (identity,)).fetchone()["id"]
        sets_seen = set()
        
        for entry in entries:
            set_slug = entry["set_slug"]
            source_image = entry["image_path"]
            filename = entry["filename"]
            name_stem = Path(filename).stem
            
            if set_slug not in sets_seen:
                db.execute("INSERT OR IGNORE INTO sets (persona_id, slug) VALUES (?, ?)", (persona_id, set_slug))
                sets_seen.add(set_slug)
                
            set_id = db.execute("SELECT id FROM sets WHERE persona_id = ? AND slug = ?", (persona_id, set_slug)).fetchone()["id"]
            
            face_dir = faces_dir / identity / set_slug
            if not face_dir.exists():
                continue
                
            face_files = sorted(f for f in face_dir.iterdir() if f.name.startswith(f"{name_stem}_face") and f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            
            if verbose and face_files:
                print(f"[{identity}/{set_slug}] Found {len(face_files)} face(s) for {filename}")
                
            for ff in face_files:
                face_name = ff.stem
                try:
                    face_index = int(face_name.rsplit("_face", 1)[1])
                except:
                    face_index = 1
                    
                relative_path = str(ff.relative_to(faces_dir.parent))
                db.execute(
                    "INSERT OR IGNORE INTO images (persona_id, set_id, image_path, source_image, face_index, status) VALUES (?, ?, ?, ?, ?, 'unreviewed')",
                    (persona_id, set_id, relative_path, source_image, face_index)
                )
                changed = db.execute("SELECT changes()").fetchone()[0]
                total += changed
                
                if verbose:
                    if changed > 0:
                        print(f"  -> INSERTED: {relative_path} (face {face_index})")
                    else:
                        print(f"  -> SKIPPED (already exists): {relative_path} (face {face_index})")
                
    db.commit()
    db.close()
    return total
