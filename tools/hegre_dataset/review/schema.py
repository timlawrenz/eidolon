import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER NOT NULL,
    slug TEXT NOT NULL,
    FOREIGN KEY (persona_id) REFERENCES personas(id),
    UNIQUE(persona_id, slug)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id INTEGER NOT NULL,
    set_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    source_image TEXT NOT NULL,
    face_index INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_at TEXT,
    FOREIGN KEY (persona_id) REFERENCES personas(id),
    FOREIGN KEY (set_id) REFERENCES sets(id),
    UNIQUE(persona_id, set_id, source_image, face_index)
);

CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_persona ON images(persona_id);
"""

def get_db(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    db.commit()
    return db
