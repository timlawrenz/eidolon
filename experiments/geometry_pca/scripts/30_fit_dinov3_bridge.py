#!/usr/bin/env python3
"""
Phase 3: DINOv3 Bridge Premise Gate (FFHQ).

Fits Ridge(X=dinov3_cls -> Y=z_x) via 5-fold CV.
Outputs variance-weighted held-out R² and per-component spectrum.
Gate: Variance-weighted R² >= 0.5, and C1-C10 individually >= 0.6.
Also runs permutation null (shuffled pairs) and MLP diagnostic.
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "output"
ALPHAS = np.logspace(-2, 4, 7)
K_FOLDS = 5


def get_variance_weights(encoder_path):
    enc = dict(np.load(encoder_path))
    evr = enc["explained_variance_ratio"]
    # normalize so weights sum to 1
    return (evr / evr.sum()).astype(np.float64)


def fit_bridge(X, Y, weights, name):
    print(f"\n{'='*60}\n  Fitting Bridge: {name} (N={len(X)}, D={Y.shape[1]})\n{'='*60}")
    t0 = time.time()
    
    # 5-fold CV for held-out R^2
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    Y_pred = np.zeros_like(Y)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_te = X[test_idx]
        
        # RidgeCV automatically selects best alpha via internal generalized CV
        model = RidgeCV(alphas=ALPHAS, store_cv_results=False)
        model.fit(X_tr, Y_tr)
        Y_pred[test_idx] = model.predict(X_te)
        print(f"  Fold {fold+1}/{K_FOLDS} done  (alpha picked: {model.alpha_})")

    # Metrics on the out-of-fold predictions
    # R^2_i = 1 - MSE_i / VAR_i. Since targets are whitened, VAR_i ~ 1.
    mse_per_comp = np.mean((Y - Y_pred)**2, axis=0)
    var_per_comp = np.var(Y, axis=0)
    r2_per_comp = 1.0 - (mse_per_comp / var_per_comp)
    
    # Variance-weighted mean R^2
    weighted_r2 = np.sum(r2_per_comp * weights)
    
    # C1-C10 gate
    c1_c10_min = np.min(r2_per_comp[:10])
    
    print(f"\n  Variance-weighted held-out R^2 : {weighted_r2:.4f}")
    print(f"  C1-C10 minimum R^2           : {c1_c10_min:.4f}")
    
    # Permutation null (sanity check)
    rng = np.random.default_rng(123)
    Y_shuffled = Y.copy()
    rng.shuffle(Y_shuffled)
    null_mse = np.mean((Y_shuffled - Y_pred)**2, axis=0)
    null_r2 = np.sum((1.0 - (null_mse / var_per_comp)) * weights)
    print(f"  Permutation null R^2           : {null_r2:.4f}")

    # Full fit on all data to save weights
    print(f"\n  Refitting on full dataset to save weights...")
    final_model = RidgeCV(alphas=ALPHAS).fit(X, Y)
    
    res = {
        "weighted_r2": float(weighted_r2),
        "c1_c10_min_r2": float(c1_c10_min),
        "r2_spectrum": r2_per_comp.tolist(),
        "null_r2": float(null_r2),
        "best_alpha": float(final_model.alpha_ if final_model.alpha_ is not None else 0.0),
        "W_coef": np.asarray(final_model.coef_).astype(np.float32),
        "W_intercept": np.asarray(final_model.intercept_).astype(np.float32)
    }
    print(f"  Done in {time.time()-t0:.1f}s")
    return res


def run_mlp_diagnostic(X, Y, weights):
    print("\n  [Diagnostic] Running MLP probe (0.25 <= R2 < 0.5 band)...")
    # Single fold (80/20) for speed
    rng = np.random.default_rng(99)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[idx[:split]], Y[idx[:split]]
    X_te, Y_te = X[idx[split:]], Y[idx[split:]]
    
    mlp = MLPRegressor(hidden_layer_sizes=(512, 128), max_iter=20, random_state=1)
    mlp.fit(X_tr, Y_tr)
    Y_pred = mlp.predict(X_te)
    
    mse_per_comp = np.mean((Y_te - Y_pred)**2, axis=0)
    var_per_comp = np.var(Y_te, axis=0)
    r2_per_comp = 1.0 - (mse_per_comp / var_per_comp)
    weighted_r2 = np.sum(r2_per_comp * weights)
    print(f"  MLP Variance-weighted held-out R^2: {weighted_r2:.4f}")
    return float(weighted_r2)


def main():
    d = np.load("data/bridge_dataset.npz")
    X_dino = d["X_dino"]
    Y_zg = d["Y_zg"]
    Y_za = d["Y_za"]
    
    w_g = get_variance_weights("output/encoder_production.npz")
    w_a = get_variance_weights("output/encoder_za_xy.npz")
    
    results = {}
    
    # 1. Fit z_g
    res_g = fit_bridge(X_dino, Y_zg, w_g, "z_g (Geometry)")
    if 0.25 <= res_g["weighted_r2"] < 0.5:
        res_g["mlp_r2"] = run_mlp_diagnostic(X_dino, Y_zg, w_g)
    results["z_g"] = res_g
        
    # 2. Fit z_a
    res_a = fit_bridge(X_dino, Y_za, w_a, "z_a_xy (Surface)")
    if 0.25 <= res_a["weighted_r2"] < 0.5:
        res_a["mlp_r2"] = run_mlp_diagnostic(X_dino, Y_za, w_a)
    results["z_a_xy"] = res_a
    
    # Verdicts
    print(f"\n{'='*60}\n  PHASE 3 VERDICT (FFHQ Premise Gate)\n{'='*60}")
    
    for k in ["z_g", "z_a_xy"]:
        r = results[k]
        pass_gate = r["weighted_r2"] >= 0.5 and r["c1_c10_min_r2"] >= 0.6
        verdict = "PASS" if pass_gate else "FAIL"
        print(f"  {k:8s} : {verdict}  (w-R2: {r['weighted_r2']:.3f} | C1-C10 min: {r['c1_c10_min_r2']:.3f})")

    # Save artifacts
    np.savez_compressed("output/bridge_dinov3.npz",
                        W_g_coef=res_g["W_coef"], W_g_intercept=res_g["W_intercept"],
                        W_a_coef=res_a["W_coef"], W_a_intercept=res_a["W_intercept"])
    
    # Strip heavy arrays before saving json
    for k in ["z_g", "z_a_xy"]:
        del results[k]["W_coef"]
        del results[k]["W_intercept"]
        
    with open("data/phase3_bridge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved output/bridge_dinov3.npz and data/phase3_bridge_results.json")

if __name__ == "__main__":
    main()
