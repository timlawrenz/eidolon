import pytest
import sqlite3
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_pca.data_loader import get_hegre_cross_shoot_paths, prepare_cross_shoot_split

@pytest.fixture
def mock_hegre_dir(tmp_path):
    db_path = tmp_path / "review.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE personas (id INTEGER PRIMARY KEY, name TEXT)
    """)
    c.execute("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            persona_id INTEGER,
            set_id INTEGER,
            image_path TEXT,
            status TEXT
        )
    """)
    
    # Insert persona names
    c.executemany("INSERT INTO personas VALUES (?, ?)", [(1, 'p1'), (2, 'p2')])
    
    # Real path layout: 
    # DB image_path: faces/p1/s1/img1.jpg
    # T5 (PREFERRED): stratum/faces/p1/s1/img1/t5_hidden.npy (faces/ prefix kept)
    # T5 (FALLBACK):  stratum/p1/s1/img1/t5_hidden.npy (old Phase 5a tree, without faces/)
    # AF: auraface/faces/p1/s1/img1.npy (faces/ kept, .jpg→.npy)
    rows = [
        (1, 1, 1, "faces/p1/s1/img1.jpg", "approved"),
        (2, 1, 1, "faces/p1/s1/img2.jpg", "approved"),
        (3, 1, 2, "faces/p1/s2/img3.jpg", "approved"),
        (4, 1, 2, "faces/p1/s2/img4.jpg", "approved"),
        (5, 2, 3, "faces/p2/s3/img5.jpg", "approved"),
        (6, 2, 3, "faces/p2/s3/img6.jpg", "unreviewed"),
    ]
    c.executemany("INSERT INTO images VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    
    # Create files matching real layout (preferred: faces/ prefix kept)
    for img_id, t5_rel_preferred, t5_rel_fallback, af_rel, t5_valid, af_valid in [
        ("img1", "faces/p1/s1/img1", "p1/s1/img1", "faces/p1/s1/img1.jpg", True, True),
        ("img2", "faces/p1/s1/img2", "p1/s1/img2", "faces/p1/s1/img2.jpg", False, True),
        ("img3", "faces/p1/s2/img3", "p1/s2/img3", "faces/p1/s2/img3.jpg", True, False),
        ("img4", "faces/p1/s2/img4", "p1/s2/img4", "faces/p1/s2/img4.jpg", True, True),
        ("img5", "faces/p2/s3/img5", "p2/s3/img5", "faces/p2/s3/img5.jpg", True, True),
        ("img6", "faces/p2/s3/img6", "p2/s3/img6", "faces/p2/s3/img6.jpg", True, True),
    ]:
        if t5_valid:
            p = tmp_path / "stratum" / t5_rel_preferred / "t5_hidden.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        if af_valid:
            p = tmp_path / "auraface" / af_rel.replace('.jpg', '.npy')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            
    return tmp_path

def test_get_hegre_cross_shoot_paths(mock_hegre_dir):
    data = get_hegre_cross_shoot_paths(mock_hegre_dir / "review.db", mock_hegre_dir)
    
    # Persona 1 should have Set 1 (img1 only) and Set 2 (img4 only)
    assert 1 in data
    assert 1 in data[1]
    assert 2 in data[1]
    assert len(data[1][1]) == 1
    assert "img1" in str(data[1][1][0]["t5_path"])
    assert len(data[1][2]) == 1
    assert "img4" in str(data[1][2][0]["auraface_path"])
    
    # Persona 2 should have Set 3 (img5 only, img6 unreviewed)
    assert 2 in data
    assert 3 in data[2]
    assert len(data[2][3]) == 1
    assert "img5" in str(data[2][3][0]["t5_path"])

def test_prepare_cross_shoot_split():
    data = {
        1: {
            1: [{"t5_path": "t1_s1", "auraface_path": "a1_s1"}],
            2: [{"t5_path": "t1_s2", "auraface_path": "a1_s2"}],
        },
        2: {
            3: [{"t5_path": "t2_s3", "auraface_path": "a2_s3"}],
        },
        3: {
            4: [{"t5_path": "t3_s4", "auraface_path": "a3_s4"}],
            5: [{"t5_path": "t3_s5", "auraface_path": "a3_s5"}],
            6: [{"t5_path": "t3_s6", "auraface_path": "a3_s6"}],
        }
    }
    query_items, index_items = prepare_cross_shoot_split(data, min_sets=2, seed=42)
    assert len(query_items) > 0
    query_personas = set([item["persona_id"] for item in query_items])
    assert query_personas == {1, 3}
    query_t5s = set([item["t5_path"] for item in query_items])
    index_t5s = set([item["t5_path"] for item in index_items])
    assert len(query_t5s.intersection(index_t5s)) == 0
    p2_index = [item for item in index_items if item["persona_id"] == 2]
    assert len(p2_index) == 1

def test_auraface_to_lda():
    from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda
    assert callable(clean_auraface)
    assert callable(project_to_lda)

