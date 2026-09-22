import os, sys, random
from pathlib import Path
from collections import defaultdict


def get_hegre_cross_shoot_paths(db_path: Path, root_dir: Path, persona_names: list = None) -> dict:
    """
    Query review dataset for approved images and return valid T5 and AuraFace paths.
    Uses HegreDataset (PostgreSQL) — db_path is the dataset root, not a .db file.
    
    Path mapping (both keep faces/ prefix):
      DB image_path:  faces/adriana/adriana-introduction/img.jpg
      T5:             stratum/faces/adriana/adriana-introduction/img/t5_hidden.npy
      AuraFace:       auraface/faces/adriana/adriana-introduction/img.npy
    
    Returns: {persona_id: {set_id: [{"t5_path": Path, "auraface_path": Path}, ...]}}
    """
    # Ensure project root is on sys.path for tools.hegre_dataset import
    _proj_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_proj_root) not in sys.path:
        sys.path.insert(0, str(_proj_root))
    
    from tools.hegre_dataset.dataset import HegreDataset
    os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')
    
    ds = HegreDataset(root_dir)
    
    if persona_names:
        name_to_id = {p.name.lower(): p.id for p in ds.personas.values()}
        valid_ids = [pid for name, pid in name_to_id.items() if name in [n.lower() for n in persona_names]]
        if not valid_ids:
            return {}
        placeholders = ','.join(['?'] * len(valid_ids))
        query = f"""
            SELECT i.persona_id, i.set_id, i.image_path 
            FROM images i
            WHERE i.status = 'approved' AND i.persona_id IN ({placeholders})
        """
        rows = ds.db.execute(query, valid_ids).fetchall()
    else:
        rows = ds.db.execute("""
            SELECT persona_id, set_id, image_path 
            FROM images WHERE status = 'approved'
        """).fetchall()
    
    data = defaultdict(lambda: defaultdict(list))
    
    for persona_id, set_id, img_path in rows:
        rel = img_path.replace('.jpg', '')
        t5_path = root_dir / 'stratum' / rel / 't5_hidden.npy'
        af_path = root_dir / 'auraface' / img_path.replace('.jpg', '.npy')
        
        if t5_path.exists() and af_path.exists():
            data[persona_id][set_id].append({
                "t5_path": t5_path,
                "auraface_path": af_path
            })
            
    return {p_id: dict(sets) for p_id, sets in data.items()}


def prepare_cross_shoot_split(data_dict: dict, min_sets: int = 2, seed: int = 42):
    """
    For each persona with >= min_sets T5 sets, hold out ONE set as query.
    All other sets (and all sets of personas with < min_sets) go to index.
    """
    rng = random.Random(seed)
    query_items, index_items = [], []
    
    for persona_id, sets in data_dict.items():
        set_ids = list(sets.keys())
        if len(set_ids) >= min_sets:
            query_set_id = rng.choice(set_ids)
            for set_id, items in sets.items():
                for item in items:
                    item_copy = dict(item)
                    item_copy["persona_id"] = persona_id
                    item_copy["set_id"] = set_id
                    if set_id == query_set_id:
                        query_items.append(item_copy)
                    else:
                        index_items.append(item_copy)
        else:
            for set_id, items in sets.items():
                for item in items:
                    item_copy = dict(item)
                    item_copy["persona_id"] = persona_id
                    item_copy["set_id"] = set_id
                    index_items.append(item_copy)
                    
    return query_items, index_items