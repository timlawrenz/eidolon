#!/usr/bin/env python3
"""
Phase 3b Correction: Face-cropped DINOv3 Transfer Gate.

Evaluates the FFHQ-fit linear bridge (W) on hegre FACE CROPS (dinov3_cls_face.npy),
fixing the domain mismatch where W was trained on FFHQ face-crops but tested on
hegre full-scene embeddings.

Outputs the true identity transfer AUC, plus the critical baseline:
AUC(raw dinov3_cls_face).
"""
import os, sys, json, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc, partition_gate

CONF_THRESH = 0.5
SEEDS = [0, 1, 2]
EPS = 0.01

def load_bridge():
    path = "output/bridge_dinov3.npz"
    if not os.path.exists(path):
        print(f"FATAL: Bridge weights not found at {path}.")
        sys.exit(1)
    d = dict(np.load(path))
    return {
        "W_g": d["W_g_coef"], "b_g": d["W_g_intercept"],
        "W_a": d["W_a_coef"], "b_a": d["W_a_intercept"]
    }

def extract_hegre_dino_face():
    db = sqlite3.connect("file:data/review.db?mode=ro", uri=True)
    c = db.cursor()
    c.execute("""
        SELECT i.enriched_dir, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
        ORDER BY p.name, i.id
    """)
    rows = c.fetchall()
    db.close()

    X_dino = []
    y_labels = []
    y_names = []
    n_skip = 0

    for ed_rel, name, pid in rows:
        ed_abs = os.path.join(
            "/mnt/nas-ai-models/training-data/eidolon/hegre_enriched",
            ed_rel.split("hegre_enriched/", 1)[1]
        )
        try:
            dino = np.load(os.path.join(ed_abs, "dinov3_cls_face.npy")).astype(np.float32)
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
    print(f"Extracted {len(X_dino)} face images across {len(np.unique(y_arr))} identities (skipped {n_skip})")
    return X_dino, y_arr, np.array(y_names)

def main():
    print("=" * 60)
    print("  PHASE 3b CORRECTION (Face-Cropped Identity Transfer Gate)")
    print("=" * 60)

    bridge = load_bridge()
    X_dino, y, names = extract_hegre_dino_face()
    if len(X_dino) == 0:
        print("No valid face images found. Wait for scripts/33 to finish.")
        sys.exit(1)

    # 1. Baseline: Raw DINOv3 Face Token Identity
    print("\n[Baseline] Raw DINOv3 Face Token Identity:")
    auc_raw_seeds = [verification_auc(X_dino, y, seed=s)[0] for s in SEEDS]
    auc_raw_mean = float(np.mean(auc_raw_seeds))
    print(f"  AUC(raw dinov3_cls_face) : {auc_raw_mean:.4f}  (per-seed: {[f'{a:.4f}' for a in auc_raw_seeds]})")

    # 2. Predicted Sliders: Ŷ = W * dinov3_cls_face + b
    Y_pred_g = (X_dino @ bridge["W_g"].T) + bridge["b_g"]
    Y_pred_a = (X_dino @ bridge["W_a"].T) + bridge["b_a"]

    # 3. Gate tests
    print("\n[Bridge Transfer] Predicted Sliders Identity:")
    auc_g_seeds = [verification_auc(Y_pred_g, y, seed=s)[0] for s in SEEDS]
    auc_g_mean = float(np.mean(auc_g_seeds))
    print(f"  Ŷ_g (Predicted Geometry) AUC : {auc_g_mean:.4f}")

    auc_a_seeds = [verification_auc(Y_pred_a, y, seed=s)[0] for s in SEEDS]
    auc_a_mean = float(np.mean(auc_a_seeds))
    print(f"  Ŷ_a (Predicted Surface)  AUC : {auc_a_mean:.4f}")

    deltas = [partition_gate(Y_pred_g, Y_pred_a, y, eps=EPS, seed=s)["delta"] for s in SEEDS]
    delta_mean = float(np.mean(deltas))
    print(f"  [Ŷ_g|Ŷ_a] AUC delta vs Ŷ_g : {delta_mean:+.4f}")

    # Random Projection Null (The Control that killed 31_evaluate)
    print("\n[Control] Random 50-D Projection of DINO Face Token:")
    proj_aucs = []
    for ps in range(5):
        prng = np.random.default_rng(1000 + ps)
        P = prng.normal(size=(1024, 50)).astype(np.float32) / np.sqrt(1024)
        Z = X_dino @ P
        proj_aucs.append(float(np.mean([verification_auc(Z, y, seed=s)[0] for s in SEEDS])))
    null_mean = float(np.mean(proj_aucs))
    print(f"  AUC(random_proj) : {null_mean:.4f} ± {np.std(proj_aucs):.4f}")

    # Final Verdict
    print("\n" + "=" * 60)
    verdict = "PASS" if auc_a_mean > null_mean and auc_a_mean > 0.51 else "FAIL"
    print(f"  GATE VERDICT: {verdict} (Must beat random proj {null_mean:.4f})")

    results = {
        "auc_raw_dino_face": auc_raw_mean,
        "auc_pred_g": auc_g_mean,
        "auc_pred_a": auc_a_mean,
        "auc_random_proj_null": null_mean,
        "verdict": verdict
    }
    with open("data/phase3b_face_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
