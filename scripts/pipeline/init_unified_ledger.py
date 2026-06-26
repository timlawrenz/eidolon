import sqlite3
import os
from pathlib import Path
import subprocess

def setup():
    # 1. NAS Directories
    nas_root = Path("/mnt/nas-ai-models/training-data/eidolon/unified_cache")
    dirs = ["aligned", "auraface", "pg"]
    for d in dirs:
        for source in ["hegre", "ffhq"]:
            (nas_root / d / source).mkdir(parents=True, exist_ok=True)
            
    # 2. Local Symlink
    local_data = Path("data/unified_cache")
    if not local_data.exists():
        local_data.symlink_to(nas_root)
        print(f"Created symlink: {local_data} -> {nas_root}")

    # 3. Gitignore
    gitignore = Path(".gitignore")
    ignores = ["data/unified_cache", "data/unified_state.db"]
    if gitignore.exists():
        content = gitignore.read_text()
        with open(gitignore, "a") as f:
            for ignore in ignores:
                if ignore not in content:
                    f.write(f"\n{ignore}\n")
                    print(f"Added {ignore} to .gitignore")

    # 4. SQLite Ledger
    db_path = Path("data/unified_state.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL, -- 'hegre' or 'ffhq'
            rel_path TEXT NOT NULL, -- The deterministic relative path (e.g., 'persona/set/img.jpg')
            stratum_done BOOLEAN DEFAULT FALSE,
            aligned_done BOOLEAN DEFAULT FALSE,
            auraface_done BOOLEAN DEFAULT FALSE,
            pg_done BOOLEAN DEFAULT FALSE,
            UNIQUE(source, rel_path)
        );
        CREATE INDEX IF NOT EXISTS idx_source ON pipeline_state(source);
        CREATE INDEX IF NOT EXISTS idx_status ON pipeline_state(stratum_done, aligned_done, auraface_done, pg_done);
    """)
    conn.commit()
    conn.close()
    print(f"Initialized SQLite ledger at {db_path}")

if __name__ == "__main__":
    setup()
