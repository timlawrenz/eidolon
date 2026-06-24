#!/usr/bin/env python3
"""
Phase 3b: DINOv3 Bridge Transfer Gate (hegre).

Evaluates whether the FFHQ-fit linear bridge (W) preserves identity signal when
applied out-of-domain to hegre editorial photos.
Computes predicted sliders: Ŷ = W * dinov3_cls
Runs canonical verification AUC on Ŷ_g and [Ŷ_g|Ŷ_a].
"""
import os, sys, json, sqlite3, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc, partition_gate

CONF_THRESH = 0.5
SEEDS = [0, 1, 2]
EPS = 0.01


def load_bridge():
    """Load the FFHQ-fit ridge regression weights."""
    path = "output/bridge_dinov3.npz"
    if not os.path.exists(path):
        print(f"FATAL: Bridge weights not found at {path}. Run scripts/30 first.")
        sys.exit(1)
    d = dict(np.load(path))
    return {
        "W_g": d["W_g_coef"], "b_g": d["W_g_intercept"],
        "W_a": d["W_a_coef"], "b_a": d["W_a_intercept"]
    }


def extract_hegre_dino(limit=0):
    """READ-ONLY query of review.db to get dinov3 tokens for clean images."""
    db = sqlite3.connect("file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro", uri=True)
    c = db.cursor()
    c.execute("""
        SELECT i.image_path, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
        ORDER BY p.name, i.id
    """)
    all_rows = c.fetchall()
    db.close()

    id_rows = {}
    for img_path, name, pid in all_rows:
        id_rows.setdefault(pid, []).append((img_path, name))
    identities = list(id_rows.items())
    if limit > 0:
        identities = identities[:limit]

    X_dino = []
    y_labels = []
    y_names = []
    n_skip = 0

    for pid, rows in identities:
        for img_path, name in rows:
            try:
                # enriched_dir is relative (data/hegre_enriched/...), but the
                # dinov3 stratum pass writes to the ABSOLUTE NAS path.
                # Normalize to the correct absolute location.
                stem = os.path.splitext(img_path[len("faces/"):] if img_path.startswith("faces/") else img_path)[0]
                ed_abs = os.path.join("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum", stem)
                
                dinov3_path = os.path.join(ed_abs, "dinov3_cls.npy")
                dino = np.load(dinov3_path).astype(np.float32)
                pose = np.load(os.path.join(ed_abs, "pose.npy")).astype(np.float32)
                if pose[23:91, 2].mean() < CONF_THRESH:
                    n_skip += 1
                    continue
            except (FileNotFoundError, OSError, ValueError):
                n_skip += 1
                continue
            
            X_dino.append(dino)
            y_labels.append(pid)
            y_names.append(name)

    X_dino = np.stack(X_dino) if X_dino else np.zeros((0, 1024), dtype=np.float32)
    y_arr = np.array(y_labels, dtype=np.int32)
    print(f"Extracted {len(X_dino)} images across {len(np.unique(y_arr))} identities (skipped {n_skip})")
    return X_dino, y_arr, np.array(y_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N identities")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHASE 3b VERDICT (hegre Identity Transfer Gate)")
    print("=" * 60)

    bridge = load_bridge()
    print("Loading hegre dinov3_cls tokens...")
    X_dino, y, names = extract_hegre_dino(args.limit)
    if len(X_dino) == 0:
        print("No valid images found. Is the stratum dinov3 pass running?")
        sys.exit(1)

    # 1. Predict the sliders: Ŷ = W * dinov3_cls + b
    Y_pred_g = (X_dino @ bridge["W_g"].T) + bridge["b_g"]
    Y_pred_a = (X_dino @ bridge["W_a"].T) + bridge["b_a"]

    # 2. Gate 3b tests
    # z_g baseline
    auc_g_seeds = [verification_auc(Y_pred_g, y, seed=s)[0] for s in SEEDS]
    auc_g_mean = float(np.mean(auc_g_seeds))
    print(f"\n  Ŷ_g (Predicted Geometry) AUC : {auc_g_mean:.4f}  (per-seed: {[f'{a:.4f}' for a in auc_g_seeds]})")

    # z_a alone
    auc_a_seeds = [verification_auc(Y_pred_a, y, seed=s)[0] for s in SEEDS]
    auc_a_mean = float(np.mean(auc_a_seeds))
    print(f"  Ŷ_a (Predicted Surface)  AUC : {auc_a_mean:.4f}  (per-seed: {[f'{a:.4f}' for a in auc_a_seeds]})")

    # concatenated
    deltas = [partition_gate(Y_pred_g, Y_pred_a, y, eps=EPS, seed=s)["delta"] for s in SEEDS]
    delta_mean = float(np.mean(deltas))
    print(f"  [Ŷ_g|Ŷ_a] AUC delta vs Ŷ_g : {delta_mean:+.4f}  (per-seed: {[f'{d:+.4f}' for d in deltas]})")

    # The actual gate condition: does Ŷ_a transfer identity > 0.51?
    verdict = "PASS" if auc_a_mean > 0.51 else "FAIL"
    print(f"\n  GATE 3b VERDICT: {verdict}  (Ŷ_a AUC > 0.51)")
    
    # Secondary: fraction of real z_a lift retained
    # real z_a AUC on this same dataset was 0.562 (chance=0.5, lift=0.062)
    # predicted lift = auc_a_mean - 0.5
    fraction = max(0.0, (auc_a_mean - 0.5)) / 0.062
    print(f"  Identity lift retained     : {fraction*100:.1f}%  (vs real z_a AUC 0.562)")

    results = {
        "n_images": len(X_dino),
        "n_identities": len(np.unique(y)),
        "Y_g_auc_3seed_mean": auc_g_mean,
        "Y_a_auc_3seed_mean": auc_a_mean,
        "cat_delta_3seed_mean": delta_mean,
        "verdict": verdict,
        "fraction_identity_retained": float(fraction)
    }

    with open("data/phase3b_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved data/phase3b_transfer_results.json")


if __name__ == "__main__":
    main()
