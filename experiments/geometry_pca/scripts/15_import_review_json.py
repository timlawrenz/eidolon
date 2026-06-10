#!/usr/bin/env python3
"""Import existing review decisions from gate_review.json into the SQLite DB."""
import json, sqlite3

DB = "data/review.db"
REVIEW = "data/gate_review.json"

def main():
    db = sqlite3.connect(DB)
    review = json.load(open(REVIEW))
    
    for name, info in review["identities"].items():
        pid_row = db.execute("SELECT id FROM personas WHERE name=?", (name,)).fetchone()
        if not pid_row:
            print(f"  SKIP {name}: not in DB")
            continue
        pid = pid_row[0]
        
        verdict = info["verdict"]
        if verdict == "CONTAMINATED":
            db.execute(
                "UPDATE images SET status='tainted:contamination', reviewed_at=datetime('now') WHERE persona_id=?",
                (pid,)
            )
            print(f"  {name}: -> CONTAMINATED (all {db.execute('SELECT changes()').fetchone()[0]} images)")
        elif verdict == "CLEAN":
            # Mark all as approved
            db.execute(
                "UPDATE images SET status='approved', reviewed_at=datetime('now') WHERE persona_id=?",
                (pid,)
            )
            n = db.execute("SELECT changes()").fetchone()[0]
            
            # Then taint specific extraction failures
            failures = info.get("extraction_failures", [])
            if failures:
                # We can't map grid positions back to specific images easily here.
                # Instead, taint any images from sets that had high failure rates.
                # For now, just log that failures exist.
                print(f"  {name}: -> CLEAN ({n} imgs, {len(failures)} extraction failures noted)")
            else:
                print(f"  {name}: -> CLEAN ({n} imgs)")
        elif verdict == "INSUFFICIENT":
            db.execute(
                "UPDATE images SET status='tainted:insufficient', reviewed_at=datetime('now') WHERE persona_id=?",
                (pid,)
            )
            print(f"  {name}: -> INSUFFICIENT")
    
    db.commit()
    
    # Stats
    for status in ['unreviewed', 'approved', 'tainted:contamination', 'tainted:insufficient']:
        n = db.execute("SELECT COUNT(*) FROM images WHERE status=?", (status,)).fetchone()[0]
        print(f"  {status}: {n}")
    
    db.close()

if __name__ == "__main__":
    main()
