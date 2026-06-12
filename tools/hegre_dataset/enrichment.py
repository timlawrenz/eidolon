import sqlite3
import subprocess
from pathlib import Path

def generate_approved_list(dataset_dir: Path) -> Path:
    db_path = dataset_dir / "review.db"
    faces_dir = dataset_dir / "faces"
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT image_path FROM images WHERE status = 'approved'")
    rows = c.fetchall()
    conn.close()
    
    approved_paths = []
    for row in rows:
        image_path = row[0]
        # Paths in DB are relative to faces_dir
        abs_path = (faces_dir / image_path).absolute()
        approved_paths.append(str(abs_path))
        
    out_file = dataset_dir / "stratum_approved_list.txt"
    out_file.write_text("\n".join(approved_paths))
    return out_file

def run_stratum_enrichment(list_path: Path, dataset_dir: Path):
    output_dir = dataset_dir / "stratum"
    
    cmd = [
        "stratum",
        "process",
        "--passes", "pose,seg,depth,normal",
        "--image-list", str(list_path),
        "--output", str(output_dir)
    ]
    
    subprocess.run(cmd, check=True)
