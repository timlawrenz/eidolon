import sys
import os
import sqlite3
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_flame import get_flame_targets

def test_get_flame_targets():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "review.db")
        img_root = os.path.join(td, "images")
        stratum_root = os.path.join(td, "stratum")
        
        # Setup DB
        db = sqlite3.connect(db_path)
        db.execute("CREATE TABLE personas (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE images (persona_id INTEGER, image_path TEXT, status TEXT)")
        db.execute("INSERT INTO personas VALUES (1, 'p1')")
        
        # Good image (exists)
        db.execute("INSERT INTO images VALUES (1, 'faces/p1/img1.jpg', 'approved')")
        # Bad image (file missing)
        db.execute("INSERT INTO images VALUES (1, 'faces/p1/img2.jpg', 'approved')")
        
        db.commit()
        db.close()
        
        # Setup files
        os.makedirs(os.path.join(img_root, "faces/p1"), exist_ok=True)
        open(os.path.join(img_root, "faces/p1/img1.jpg"), "w").close()
        
        # We need pose.npy from stratum to get the face crop bounding box
        s1 = os.path.join(stratum_root, "p1/img1")
        os.makedirs(s1, exist_ok=True)
        open(os.path.join(s1, "pose.npy"), "w").close()
        
        targets = get_flame_targets(f"file:{db_path}?mode=ro&nolock=1", stratum_root, img_root)
        
        assert len(targets) == 1
        assert targets[0][0] == os.path.join(img_root, "faces/p1/img1.jpg")
        assert targets[0][1] == os.path.join(stratum_root, "p1/img1/pose.npy")
        assert targets[0][2] == 1

if __name__ == "__main__":
    test_get_flame_targets()
    print("Tests passed!")
