"""One-shot SQLite → PostgreSQL data export and verification.

The export produces PostgreSQL-compatible INSERT SQL. The verification
compares row counts and numeric values between the two databases.

Exported columns exclude `auraface_embedding` (populated later by `enrich`).
"""

import sqlite3
from pathlib import Path

# FK dependency order: parent tables must be inserted before child tables.
TABLE_ORDER = ["personas", "sets", "images"]

# Columns to skip during export (populated by application logic, not migration).
SKIP_COLUMNS = {"auraface_embedding"}


def export_sqlite_to_sql(db_path: Path) -> str:
    """Dump entire SQLite database to PostgreSQL-compatible INSERT SQL.

    Args:
        db_path: Path to the SQLite review.db file.

    Returns:
        SQL string with one INSERT statement per row, ordered by FK dependency.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    lines = []

    for table in TABLE_ORDER:
        # Check if table exists (empty DB has no rows but tables are created)
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        if not table_exists:
            continue

        cols = [
            r["name"] for r in conn.execute(f"PRAGMA table_info({table})")
            if r["name"] not in SKIP_COLUMNS
        ]
        if not cols:
            continue

        rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
        for row in rows:
            vals = []
            for col in cols:
                v = row[col]
                if v is None:
                    vals.append("NULL")
                elif isinstance(v, (int, float)):
                    vals.append(str(v))
                else:
                    # Escape single quotes for SQL
                    escaped = str(v).replace("'", "''")
                    vals.append(f"'{escaped}'")
            lines.append(
                f"INSERT INTO {table} ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)});"
            )

    conn.close()
    return "\n".join(lines) + "\n"


def verify_row_counts(
    sqlite_conn: sqlite3.Connection, pg_cursor
) -> list[dict]:
    """Compare row counts between SQLite and PostgreSQL for all tables.

    Args:
        sqlite_conn: sqlite3.Connection to the source database.
        pg_cursor: PostgreSQL cursor (psycopg2 or SQLAlchemy).

    Returns:
        List of dicts: {table, sqlite_count, pg_count, match}.
    """
    results = []
    for table in TABLE_ORDER:
        sl = sqlite_conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        pg_result = pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
        pg = pg_result.fetchone()[0]
        results.append({
            "table": table,
            "sqlite_count": sl,
            "pg_count": pg,
            "match": sl == pg,
        })
    return results
