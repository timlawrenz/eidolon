#!/usr/bin/env python3.11
"""Phase 5b DIAGNOSTIC: is the retrieval SPACE sound, independent of the Prior?

Ceiling control: skip the Prior. Use a GROUND-TRUTH query AuraFace (a real image
from the held-out shoot), project it the SAME way as the index, and run cross-shoot
kNN. If GT-LDA cross-shoot retrieval also fails, the retrieval space/metric is
broken (not the Prior). If GT works but Prior doesn't, the gap is the Prior.

Also tests metric/space variants to isolate the bug:
  A. raw 64-d LDA, Euclidean        (what G-A used)
  B. raw 64-d LDA, cosine
  C. z-scored 64-d LDA, Euclidean   (equalize axis variance)
  D. reconstruct->512d->L2norm, cosine  (the TRAINING metric space)
"""
import sys, time, numpy as np, random, json
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry_pca.data_loader import get_hegre_cross_shoot_paths, prepare_cross_shoot_split
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full

HEGRE_ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
DB_PATH = HEGRE_ROOT / "review.db"


def load_lda(items):
    """Load raw AuraFace, return (lda_64, raw_512, labels)."""
    lda_list, raw_list, labels = [], [], []
    for it in tqdm(items, desc="load"):
        raw = np.load(it["auraface_path"]).astype(np.float64)
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
    """LDA-64 -> reconstruct 512 -> L2 normalize (the training metric space)."""
    full = np.stack([lda_to_full(v) for v in lda_vecs])
    return full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-8)


def main():
    t0 = time.time()
    print("Loading data...")
    data = get_hegre_cross_shoot_paths(DB_PATH, HEGRE_ROOT)
    query_items, index_items = prepare_cross_shoot_split(data, min_sets=2, seed=42)
    rng = random.Random(42)
    if len(index_items) > 30000:
        index_items = rng.sample(index_items, 30000)
    print(f"Query: {len(query_items)}, Index: {len(index_items)}")

    print("Loading index LDA...")
    idx_lda, idx_raw, idx_lab = load_lda(index_items)
    print("Loading query LDA (GROUND TRUTH AuraFace from held-out shoot)...")
    q_lda, q_raw, q_lab = load_lda(query_items)

    # z-score using index statistics
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
        ("A. GT-LDA64 Euclidean (==G-A space)", q_lda, idx_lda, "euclidean"),
        ("B. GT-LDA64 cosine", q_lda, idx_lda, "cosine"),
        ("C. GT-LDA64 z-scored Euclidean", q_z, idx_z, "euclidean"),
        ("D. GT recon->512->L2norm cosine (train metric)", q_recon, idx_recon, "cosine"),
    ]
    for name, q, idx, metric in variants:
        r1 = recall_at_k(q, idx, q_lab, idx_lab, 1, metric)
        r5 = recall_at_k(q, idx, q_lab, idx_lab, 5, metric)
        r10 = recall_at_k(q, idx, q_lab, idx_lab, 10, metric)
        print(f"{name:40s} {r1:8.4f} {r5:8.4f} {r10:8.4f}")
        results[name.split('.')[0].strip()] = {"r1": r1, "r5": r5, "r10": r10}

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    out = Path("output/phase5b_space_diagnostic.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
