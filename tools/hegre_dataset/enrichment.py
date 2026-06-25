import subprocess
import shutil
from pathlib import Path
from .review.schema import get_db

def generate_approved_list(db_path: Path, faces_dir: Path, output_file: Path) -> int:
    """Query DB for approved images and write their absolute paths to output_file."""
    db = get_db(db_path)
    rows = db.execute("SELECT image_path FROM images WHERE status = 'approved'").fetchall()
    db.close()
    
    paths = []
    for row in rows:
        img_path = (faces_dir / row["image_path"]).resolve()
        paths.append(str(img_path))
        
    with open(output_file, "w") as f:
        for p in paths:
            f.write(p + "\n")
            
    return len(paths)

def run_stratum_enrichment(dataset_dir: Path, db_path: Path, faces_dir: Path):
    """Generate the approved list and invoke stratum process."""
    list_file = dataset_dir / "stratum_approved_list.txt"
    stratum_out = dataset_dir / "stratum"
    
    count = generate_approved_list(db_path, faces_dir, list_file)
    if count == 0:
        print("No approved images found. Skipping enrichment.")
        return
        
    print(f"Found {count} approved images. Invoking stratum-hq...")
    
    # Use shutil.which to detect if stratum is available
    if shutil.which("stratum") is None:
        print("Error: 'stratum' command not found in PATH.")
        print("Please ensure stratum-hq is installed (`pip install -e \".[all]\"`) and active.")
        return
    
    cmd = [
        "stratum", "process", str(faces_dir.resolve()),
        "--output", str(stratum_out.resolve()), "--passes", "pose,seg,depth,normal,caption,t5", "--image-list", str(list_file.resolve())
    ]
    subprocess.run(cmd, check=True)
