"""Tests for SQLite → PostgreSQL data migration."""
import sqlite3
import pytest
from pathlib import Path

from tools.hegre_dataset.db_migrate import export_sqlite_to_sql, verify_row_counts


def _seed_sqlite(db_path: Path):
    """Create a minimal review.db matching the real schema."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE personas (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);
        CREATE TABLE sets (id INTEGER PRIMARY KEY AUTOINCREMENT, persona_id INTEGER, slug TEXT);
        CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, persona_id INTEGER,
            set_id INTEGER, image_path TEXT, source_image TEXT, face_index INTEGER DEFAULT 1,
            status TEXT DEFAULT 'unreviewed', reviewed_at TEXT,
            zg_distance REAL, af_distance REAL);
        INSERT INTO personas VALUES (1, 'anna-l');
        INSERT INTO sets VALUES (1, 1, 'shoot1');
        INSERT INTO images VALUES (1, 1, 1, 'faces/anna-l/shoot1/a.jpg', 'a.jpg', 1, 'approved', '2024-01-01', 5.0, 0.1);
        INSERT INTO images VALUES (2, 1, 1, 'faces/anna-l/shoot1/b.jpg', 'b.jpg', 2, 'unreviewed', NULL, NULL, NULL);
    """)
    conn.commit()
    conn.close()


class TestExportSqliteToSql:
    def test_produces_valid_sql(self, tmp_path):
        """Exported SQL contains INSERT statements for all tables."""
        db_path = tmp_path / "review.db"
        _seed_sqlite(db_path)
        sql = export_sqlite_to_sql(db_path)
        assert "INSERT INTO personas" in sql
        assert "INSERT INTO sets" in sql
        assert "INSERT INTO images" in sql
        assert "anna-l" in sql
        assert "'approved'" in sql

    def test_fk_order_maintained(self, tmp_path):
        """Personas exported before sets, sets before images."""
        db_path = tmp_path / "review.db"
        _seed_sqlite(db_path)
        sql = export_sqlite_to_sql(db_path)
        pos_p = sql.find("INSERT INTO personas")
        pos_s = sql.find("INSERT INTO sets")
        pos_i = sql.find("INSERT INTO images")
        assert 0 <= pos_p < pos_s < pos_i, (
            f"FK order violated: personas={pos_p}, sets={pos_s}, images={pos_i}"
        )

    def test_null_values_exported(self, tmp_path):
        """NULL values in SQLite become SQL NULL (not empty strings)."""
        db_path = tmp_path / "review.db"
        _seed_sqlite(db_path)
        sql = export_sqlite_to_sql(db_path)
        # The unreviewed image has NULL reviewed_at, NULL zg_distance, NULL af_distance
        assert "NULL" in sql

    def test_single_quotes_escaped(self, tmp_path):
        """Single quotes in text values are escaped for SQL."""
        conn = sqlite3.connect(str(tmp_path / "review.db"))
        conn.execute("CREATE TABLE personas (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO personas VALUES (1, \"O'Brien\")")
        conn.commit()
        conn.close()
        sql = export_sqlite_to_sql(tmp_path / "review.db")
        assert "O''Brien" in sql  # escaped per SQL standard

    def test_empty_db_exports_nothing(self, tmp_path):
        """Empty database produces no INSERT statements."""
        conn = sqlite3.connect(str(tmp_path / "review.db"))
        conn.execute("CREATE TABLE personas (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()
        sql = export_sqlite_to_sql(tmp_path / "review.db")
        assert "INSERT INTO personas" not in sql


class TestVerifyRowCounts:
    def test_matching_counts(self, tmp_path):
        """verify_row_counts returns match=True when counts align."""
        _seed_sqlite(tmp_path / "review.db")
        sl = sqlite3.connect(str(tmp_path / "review.db"))

        class MockCursor:
            def execute(self, sql, params=None):
                class Result:
                    def fetchone(self):
                        return [2] if "images" in sql else [1]
                return Result()

        results = verify_row_counts(sl, MockCursor())
        assert len(results) == 3  # personas, sets, images
        assert all(r["match"] for r in results)

    def test_mismatched_counts(self, tmp_path):
        """verify_row_counts returns match=False when counts differ."""
        _seed_sqlite(tmp_path / "review.db")
        sl = sqlite3.connect(str(tmp_path / "review.db"))

        class MockCursor:
            def execute(self, sql, params=None):
                class Result:
                    def fetchone(self):
                        return [99]  # wrong for all tables
                return Result()

        results = verify_row_counts(sl, MockCursor())
        assert not any(r["match"] for r in results)
