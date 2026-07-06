#!/usr/bin/env python3
"""One-shot SQLite → PostgreSQL data migration for Hegre dataset.

Usage:
    python scripts/pg_migrate.py [--dry-run]

Exports the live review.db to PostgreSQL-compatible SQL, imports into
the database configured in config/database.yml, and verifies row counts.
"""

import sys
import sqlite3
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.hegre_dataset.db_migrate import export_sqlite_to_sql, verify_row_counts
from tools.hegre_dataset.config import database_url

# ── Configuration ──────────────────────────────────────────────────────
SQLITE_PATH = Path(
    "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
)
EXPORT_PATH = Path("/tmp/hegre_pg_export.sql")

# ── Main ────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> int:
    if not SQLITE_PATH.exists():
        print(f"ERROR: SQLite database not found at {SQLITE_PATH}", file=sys.stderr)
        print("Check the path or update SQLITE_PATH in this script.", file=sys.stderr)
        return 1

    # 1. Export
    print("1/4 Exporting SQLite to SQL...")
    sql = export_sqlite_to_sql(SQLITE_PATH)
    EXPORT_PATH.write_text(sql)
    lines = sql.count("\n")
    print(f"   Wrote {len(sql):,} bytes, {lines:,} INSERT statements")
    print(f"   Saved to {EXPORT_PATH}")

    if dry_run:
        print("\nDry run — no data imported. Export is at", EXPORT_PATH)
        return 0

    # 2. Import
    print("2/4 Importing to PostgreSQL...")
    import psycopg2
    import os

    db_url = database_url()
    # Parse psycopg2 DSN from SQLAlchemy URL
    # postgresql+psycopg2://user@host/dbname
    parts = db_url.replace("postgresql+psycopg2://", "").split("/")
    user_host = parts[0].split("@")
    user = user_host[0]
    host = user_host[1] if len(user_host) > 1 else "localhost"
    dbname = parts[1]

    pg = psycopg2.connect(host=host, user=user, dbname=dbname)
    pg.autocommit = True
    cur = pg.cursor()

    # Execute the SQL file
    with open(EXPORT_PATH) as f:
        sql_content = f.read()
    cur.execute(sql_content)
    pg.close()
    print("   Import complete.")

    # 3. Reset sequence
    print("3/4 Resetting autoincrement sequences...")
    pg = psycopg2.connect(host=host, user=user, dbname=dbname)
    pg.autocommit = True
    cur = pg.cursor()
    for table in ["personas", "sets", "images"]:
        cur.execute(
            f"SELECT setval('{table}_id_seq', COALESCE((SELECT MAX(id) FROM {table}), 1))"
        )
    pg.close()
    print("   Sequences reset.")

    # 4. Verify
    print("4/4 Verifying row counts...")
    sl = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    pg = psycopg2.connect(host=host, user=user, dbname=dbname)
    pg.autocommit = True

    results = verify_row_counts(sl, pg.cursor())
    all_match = True
    for r in results:
        status = "✓" if r["match"] else "✗ MISMATCH"
        if not r["match"]:
            all_match = False
        print(f"   {r['table']:12s} SQLite={r['sqlite_count']:6d}  PG={r['pg_count']:6d}  {status}")

    sl.close()
    pg.close()

    if all_match:
        print("\n✓ Migration verified. All row counts match.")
        print(f"\nNext steps:")
        print(f"  1. mv {SQLITE_PATH} {SQLITE_PATH}.old")
        print(f"  2. python -m tools.hegre_dataset review ui --dataset data/hegre_datasets/hegre-faces/v1")
        return 0
    else:
        print("\n✗ VERIFICATION FAILED — do not rename SQLite file.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    sys.exit(main(dry_run=dry_run))
