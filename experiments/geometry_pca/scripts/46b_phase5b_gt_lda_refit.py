#!/usr/bin/env python3
"""Phase 5b GT-LDA ceiling — REFIT 2026-07-20 with cleaned Hegre dataset.

Fast-path: queries approved images via HegreDataset (PG), no T5 filtering.
Cross-shoot split: hold out one shoot per persona (≥2 shoots), LDA-projected
AuraFace → kNN retrieval. Measures the retrieval space ceiling with refitted
LDA basis on cleaned 324-persona dataset.

Pre-registered G2: cross-shoot R@1 ≥ 0.842 (old ceiling); target ≥ 0.85.
"""
import sys, time, numpy as np, random, json, os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Setup paths
_PROJ = Path(__file__).resolve().parent.parent.parent.parent  # project root (4 levels up from scripts/)
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full

HEGRE_ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
os.environ.setdefault('EIDOLON_SKIP_REVIEWDB_GUARD', '1')


def load_cross_shoot_data():
    """Load approved images from PG, split cross-shoot (AuraFace only, no T5)."""
    from tools.hegre_dataset.dataset import HegreDataset
    ds = HegreDataset(HEGRE_ROOT)
    
    # Query approved images with set info
    rows = ds.db.execute("""
        SELECT i.persona_id, i.set_id, i.image_path
        FROM images i
        WHERE i.status = 'approved'
        ORDER BY i.persona_id, i.set_id
    """).fetchall()
    
    # Group by persona → set
    persona_sets = defaultdict(lambda: defaultdict(list))
    for pid, sid, img_path in rows:
        af_path = HEGRE_ROOT / "auraface" / img_path.replace(".jpg", ".npy")
        persona_sets[pid][sid].append({
            "auraface_path": str(af_path),
            "persona_id": pid,
            "set_id": sid
        })
    
    # Cross-shoot split: hold out ONE set per persona (≥2 sets)
    rng = random.Random(42)
    query_items, index_items = [], []
    for pid, sets in persona_sets.items():
        set_ids = list(sets.keys())
        if len(set_ids) >= 2:
            query_sid = rng.choice(set_ids)
            for sid, items in sets.items():
                if sid == query_sid:
                    query_items.extend(items)
                else:
                    index_items.extend(items)
        else:
            for items in sets.values():
                index_items.extend(items)
    
    return query_items, index_items


def load_lda(items):
    """Load raw AuraFace, return (lda_64, raw_512, labels)."""
    lda_list, raw_list, labels = [], [], []
    for it in tqdm(items, desc="load"):
        try:
            raw = np.load(it["auraface_path"]).astype(np.float64)
        except (FileNotFoundError, OSError):
            continue
        lda = project_to_lda(clean_auraface(raw)).ravel().astype(np.float32)
        lda_list.append(lda)
        raw_list.append(raw.astype(np.float32))
        labels.append(it["persona_id"])
    return np.stack(lda_list), np.stack(raw_list), labels


def recall_at_k(q, idx, q_lab, idx_lab, k, metric="euclidean"):
    d = cdist(q, idx, metric=metric)
    hits = 0
    for i, ql in enumerate(q_lab):
        top = np.argsort(d[i])[:k]
        if ql in [idx_lab[t] for t in top]:
            hits += 1
    return hits / len(q_lab)


def recon_l2(lda_vecs):
    """LDA-64 → reconstruct 512 → L2 normalize."""
    full = np.stack([lda_to_full(v) for v in lda_vecs])
    return full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-8)


def main():
    t0 = time.time()
    print("Loading cross-shoot data (AuraFace only, no T5)...")
    query_items, index_items = load_cross_shoot_data()
    rng = random.Random(42)
    if len(index_items) > 30000:
        index_items = rng.sample(index_items, 30000)
    print(f"Query: {len(query_items)}, Index: {len(index_items)}, "
          f"Personas: {len(set(i['persona_id'] for i in query_items))}")
    
    print("Loading index LDA...")
    idx_lda, idx_raw, idx_lab = load_lda(index_items)
    print("Loading query LDA (GROUND TRUTH AuraFace from held-out shoot)...")
    q_lda, q_raw, q_lab = load_lda(query_items)
    
    # z-score
    mu = idx_lda.mean(0); sd = idx_lda.std(0) + 1e-8
    idx_z = (idx_lda - mu) / sd
    q_z = (q_lda - mu) / sd
    
    # reconstructed L2 space
    idx_recon = recon_l2(idx_lda)
    q_recon = recon_l2(q_lda)
    
    chance10 = 10.0 / len(set(idx_lab))
    print(f"\nChance Recall@10 ~ {chance10:.4f}\n")
    print(f"{'variant':40s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s}")
    print("-" * 68)
    
    results = {"chance_r10": chance10, "n_query": len(query_items),
               "n_index": len(index_items), "n_personas": len(set(q_lab))}
    
    variants = [
        ("A. GT-LDA64 Euclidean", q_lda, idx_lda, "euclidean"),
        ("B. GT-LDA64 cosine", q_lda, idx_lda, "cosine"),
        ("C. GT-LDA64 z-scored Euclidean", q_z, idx_z, "euclidean"),
        ("D. GT recon->512->L2norm cosine", q_recon, idx_recon, "cosine"),
    ]
    for name, q, idx, metric in variants:
        r1 = recall_at_k(q, idx, q_lab, idx_lab, 1, metric)
        r5 = recall_at_k(q, idx, q_lab, idx_lab, 5, metric)
        r10 = recall_at_k(q, idx, q_lab, idx_lab, 10, metric)
        print(f"{name:40s} {r1:8.4f} {r5:8.4f} {r10:8.4f}")
        results[name.split('.')[0].strip()] = {"r1": r1, "r5": r5, "r10": r10}
    
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    out = Path("experiments/geometry_pca/output/phase5b_gt_lda_refit_20260720.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()