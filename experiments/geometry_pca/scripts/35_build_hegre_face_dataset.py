import sqlite3
import os
import json
from pathlib import Path

DB_URI = "file:experiments/geometry_pca/data/review.db?mode=ro"
SOURCE_ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px/"
# This data/ dir is a symlink to /mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/
DEST_ROOT = "experiments/geometry_pca/data/hegre_faces/"

def get_approved_images():
    with sqlite3.connect(DB_URI, uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM images WHERE status = 'approved'")
        return [row[0] for row in cursor.fetchall()]

if __name__ == "__main__":
    images = get_approved_images()
    print(f"Found {len(images)} approved images.")
