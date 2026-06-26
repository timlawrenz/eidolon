import sqlite3
from pathlib import Path

def ingest():
    conn = sqlite3.connect("data/unified_state.db")
    c = conn.cursor()

    # 1. Ingest Hegre (Approved only)
    print("Ingesting Hegre approved images...")
    hegre_db = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
    # Using mode=ro and nolock=1 to prevent locking the live review UI
    hegre_conn = sqlite3.connect(f"file:{hegre_db}?mode=ro&nolock=1", uri=True)
    hegre_c = hegre_conn.cursor()
    
    # Selecting the exact relative path for deterministic downstream use
    hegre_c.execute("SELECT image_path FROM images WHERE status = 'approved'")
    hegre_rows = hegre_c.fetchall()
    
    hegre_records = [("hegre", row[0]) for row in hegre_rows]
    c.executemany("""
        INSERT OR IGNORE INTO pipeline_state (source, rel_path)
        VALUES (?, ?)
    """, hegre_records)
    conn.commit()
    print(f"Ingested {c.rowcount} Hegre records.")
    hegre_conn.close()

    # 2. Ingest FFHQ
    print("Ingesting FFHQ images...")
    ffhq_raw_dir = Path("/mnt/nas-ai-models/training-data/ffhq/raw")
    ffhq_records = []
    
    # Just a sanity check to avoid hanging on 70k files, we can batch it
    # We will just yield them
    count = 0
    if ffhq_raw_dir.exists():
        for p in ffhq_raw_dir.glob("*.png"):
            # Using 'raw/00000.png' as the relative path deterministic key
            rel_path = f"raw/{p.name}"
            ffhq_records.append(("ffhq", rel_path))
            count += 1
            if count % 10000 == 0:
                c.executemany("INSERT OR IGNORE INTO pipeline_state (source, rel_path) VALUES (?, ?)", ffhq_records)
                ffhq_records = []
                print(f"  ... inserted {count} FFHQ records")
        
        if ffhq_records:
            c.executemany("INSERT OR IGNORE INTO pipeline_state (source, rel_path) VALUES (?, ?)", ffhq_records)
            print(f"  ... inserted {count} FFHQ records total")
    else:
        print("Warning: FFHQ raw dir not found.")

    conn.commit()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
