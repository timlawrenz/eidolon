#!/usr/bin/env python3.11
"""Phase 5b CORRECTED: cross-shoot Prior retrieval WITH proper T5 masking.

Fixes the conditioning bug: predict_pin now receives t5_mask and replicates
training preprocessing (valid tokens -> cap 256 -> zero-pad). Includes the
GT-LDA ceiling control on the SAME corpus snapshot for an apples-to-apples read.

Pre-registered G-A: Prior Recall@k must beat random-projection null (persona
bootstrap CI excludes 0); caption-shuffle ~ null.
"""
import sys, time, numpy as np, torch, random, json
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry_pca.poser import PoserRetrievalHarness, evaluate_recall, generate_random_null
from geometry_pca.data_loader import get_hegre_cross_shoot_paths, prepare_cross_shoot_split
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda

HEGRE_ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
DB_PATH = HEGRE_ROOT / "review.db"
PRIOR_PATH = Path("/home/tim/source/activity/eidolon/output/exp1_g2/exp1_arm_B.pt")
LDA_PATH = Path("/home/tim/source/activity/eidolon/experiments/geometry_pca/output/auraface_lda.npz")
DEVICE = "cpu"
MAX_TOKENS = 256
N_BOOT = 2000


def t5_mask_path(t5_path: Path) -> Path:
    return t5_path.parent / "t5_mask.npy"


def batched_predict_masked(harness, t5_paths, batch_size=32):
    """Load each T5 + its mask, predict pins with training-faithful masking."""
    pins = []
    for i in tqdm(range(0, len(t5_paths), batch_size), desc="Prior pins (masked)"):
        batch = t5_paths[i:i+batch_size]
        t5_stack, mask_stack = [], []
        for p in batch:
            t5 = np.load(p).astype(np.float32)            # (512,1024)
            mp = t5_mask_path(p)
            if mp.exists():
                m = np.load(mp).astype(bool)
            else:
                # fall back: treat nonzero rows as valid
                m = (np.abs(t5).sum(1) > 1e-6)
            t5_stack.append(t5)
            mask_stack.append(m)
        t5_arr = np.stack(t5_stack)
        mask_arr = np.stack(mask_stack)
        pins.extend(harness.predict_pin(t5_arr, mask=mask_arr, max_tokens=MAX_TOKENS))
    return np.array(pins)


def load_index(items):
    vecs, labels = [], []
    for it in tqdm(items, desc="index LDA"):
        raw = np.load(it["auraface_path"]).astype(np.float64)
        vecs.append(project_to_lda(clean_auraface(raw)).ravel().astype(np.float32))
        labels.append(it["persona_id"])
    return np.stack(vecs), labels


def load_query_gt_lda(items):
    """GT-LDA ceiling control: query = real held-out-shoot AuraFace -> LDA."""
    vecs = []
    for it in tqdm(items, desc="query GT-LDA"):
        raw = np.load(it["auraface_path"]).astype(np.float64)
        vecs.append(project_to_lda(clean_auraface(raw)).ravel().astype(np.float32))
    return np.stack(vecs)


def per_query_hits(pins, idx, q_lab, idx_lab, k):
    d = cdist(pins, idx, metric="euclidean")
    hits = np.zeros(len(q_lab), dtype=bool)
    for i, ql in enumerate(q_lab):
        top = np.argsort(d[i])[:k]
        hits[i] = ql in [idx_lab[t] for t in top]
    return hits


def persona_bootstrap(hits_a, hits_b, personas, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    uniq = sorted(set(personas))
    a_by = {p: hits_a[np.array([x == p for x in personas])] for p in uniq}
    b_by = {p: hits_b[np.array([x == p for x in personas])] for p in uniq}
    deltas = np.empty(n_boot)
    for bi in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        deltas[bi] = np.mean([a_by[p].mean() for p in samp]) - np.mean([b_by[p].mean() for p in samp])
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(np.mean(deltas <= 0))


def main():
    t0 = time.time()
    print("Init harness + data...")
    harness = PoserRetrievalHarness(prior_model_path=PRIOR_PATH, lda_basis_path=LDA_PATH, device=DEVICE)
    data = get_hegre_cross_shoot_paths(DB_PATH, HEGRE_ROOT)
    query_items, index_items = prepare_cross_shoot_split(data, min_sets=2, seed=42)
    rng = random.Random(42)
    if len(index_items) > 30000:
        index_items = rng.sample(index_items, 30000)
    print(f"Query: {len(query_items)}  Index: {len(index_items)}  Personas: {len(set(i['persona_id'] for i in query_items))}")

    idx_vec, idx_lab = load_index(index_items)
    harness.build_index(idx_vec, idx_lab)
    q_lab = [it["persona_id"] for it in query_items]
    q_t5 = [it["t5_path"] for it in query_items]

    print("\n--- Prior pins WITH mask ---")
    prior_pins = batched_predict_masked(harness, q_t5)
    null_pins = generate_random_null(prior_pins)
    sh_t5 = list(q_t5); random.Random(42).shuffle(sh_t5)
    shuffle_pins = batched_predict_masked(harness, sh_t5)

    print("\n--- GT-LDA ceiling (same corpus) ---")
    gt_pins = load_query_gt_lda(query_items)

    chance10 = 10.0 / len(set(idx_lab))
    print(f"\nChance R@10 ~ {chance10:.4f}\n")
    print(f"{'source':24s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s}")
    print("-" * 52)
    results = {"chance_r10": chance10, "n_query": len(query_items),
               "n_index": len(index_items), "n_personas": len(set(q_lab))}
    for name, pins in [("Prior (masked)", prior_pins), ("Random null", null_pins),
                       ("Caption-shuffle", shuffle_pins), ("GT-LDA ceiling", gt_pins)]:
        r = {k: evaluate_recall(pins, harness.index_vectors, q_lab, harness.index_labels, k) for k in (1, 5, 10)}
        print(f"{name:24s} {r[1]:8.4f} {r[5]:8.4f} {r[10]:8.4f}")
        results[name] = r

    print("\n--- Bootstrap Δ(Prior − Null) ---")
    results["bootstrap"] = {}
    for k in (1, 5, 10):
        hp = per_query_hits(prior_pins, harness.index_vectors, q_lab, harness.index_labels, k)
        hn = per_query_hits(null_pins, harness.index_vectors, q_lab, harness.index_labels, k)
        mean, lo, hi, pv = persona_bootstrap(hp, hn, q_lab)
        sig = "SIGNIFICANT" if lo > 0 else "n.s."
        print(f"  k={k:2d}: Δ={mean:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  p={pv:.4f}  {sig}")
        results["bootstrap"][f"k{k}"] = {"delta": mean, "ci_lo": lo, "ci_hi": hi, "p": pv}

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    out = Path("output/phase5b_corrected_masked.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
