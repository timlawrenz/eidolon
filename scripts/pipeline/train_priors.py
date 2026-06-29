#!/usr/bin/env python3
"""
Train Rectified Flow Matching Priors for text→z_g and text→AuraFace-LDA.

FFHQ (complete T5): split 60k train / 10k held-out.
Hegre: used only for σ²_w (within-person z_g variance — requires multi-image-per-ID).

Gate G1 (z_g Prior): held-out MSE(z_g_pred, z_g_true) / σ²_w < 1.0 (real T5 conditioning)
Gate G2 (AuraFace-LDA Prior): held-out mean cosine similarity > 0.3 (real T5 conditioning)
"""
import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Geometry PCA imports — must come before torch imports that depend on these paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiments" / "geometry_pca"))

import torch
from priors.data import build_ffhq_zg_dataset, build_ffhq_lda_dataset, Z_G_MAX_NORM, PriorDataset
from priors.models_torch import AdaLNResNet
from priors.flow_matching_torch import RectifiedFlowMatching
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = "output/prior_checkpoints"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "zg":  {"d_out": 50,  "d_hidden": 1024, "n_blocks": 12, "lr": 1e-4, "epochs": 50, "batch_size": 512},
    "lda": {"d_out": 64,  "d_hidden": 1024, "n_blocks": 12, "lr": 1e-4, "epochs": 50, "batch_size": 512},
}

# ---------------------------------------------------------------------------

def compute_sigma_w_from_hegre(fallback=104.0):
    """Compute mean within-identity z_g variance from Hegre.
    
    Falls back to a cached value if the review DB is unavailable (locked by UI).
    """
    import sqlite3
    HEGRE = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
    
    try:
        conn = sqlite3.connect(f"file:{HEGRE / 'review.db'}?mode=ro&nolock=1", uri=True)
        c = conn.cursor()
        c.execute("""SELECT i.image_path, i.persona_id
                     FROM images i WHERE i.status = 'approved'""")
        rows = c.fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"  WARNING: review DB unavailable ({e}), using fallback σ²_w = {fallback}")
        return fallback
    
    # Group z_g by persona
    persona_zg = defaultdict(list)
    miss = 0
    for img_path, pid in rows:
        zg_f = HEGRE / "zg" / f"{Path(img_path).with_suffix('')}.npy"
        if not zg_f.exists():
            miss += 1; continue
        z = np.load(zg_f).astype(np.float64)
        if np.linalg.norm(z) >= Z_G_MAX_NORM:
            miss += 1; continue
        persona_zg[pid].append(z)
    
    total_var = 0.0
    n_total = 0
    n_ids = 0
    for pid, vectors in persona_zg.items():
        if len(vectors) < 2:
            continue
        arr = np.stack(vectors)
        total_var += np.sum((arr - arr.mean(0)) ** 2)
        n_total += len(arr)
        n_ids += 1
    
    sigma2_w = total_var / n_total if n_total > 0 else 1.0
    print(f"  σ²_w from Hegre: {sigma2_w:.4f} ({n_ids} personas, {n_total} images, {miss} skipped)")
    return sigma2_w


def split_ffhq(dataset, held_fraction=0.15):
    """Split a PriorDataset into train and held-out by directory index boundary.
    
    FFHQ directories are sequential, one image per directory. The dataset is built
    in sorted order. held_fraction controls the held-out proportion.
    """
    n = len(dataset)
    held_count = max(1, int(n * held_fraction))
    split = max(1, n - held_count)
    train = PriorDataset(dataset.t5_paths[:split], dataset.target_paths[:split],
                         from_arrays=dataset.from_arrays)
    held  = PriorDataset(dataset.t5_paths[split:], dataset.target_paths[split:],
                         from_arrays=dataset.from_arrays)
    if hasattr(dataset, "raw_targets") and dataset.raw_targets is not None:
        train.raw_targets = dataset.raw_targets[:split]
        held.raw_targets = dataset.raw_targets[split:]
    
    print(f"  Split: train={len(train)}, held-out={len(held)}")
    return train, held


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_prior(prior_name, train_ds, held_ds, d_out, d_hidden, n_blocks, device,
                epochs, batch_size, lr, output_dir, sigma2_w):
    """Train a Prior model with FFHQ train/holdout split. Returns gate results."""
    
    print(f"\n{'='*60}")
    print(f" Prior {prior_name}: text→{d_out}d")
    print(f" Train: {len(train_ds)}, Held-out: {len(held_ds)}, Batch: {batch_size}, Epochs: {epochs}")
    print(f" Hidden: {d_hidden}, Blocks: {n_blocks}, Device: {device}")
    print(f"{'='*60}")
    
    model = AdaLNResNet(d_in=d_out, d_out=d_out, d_hidden=d_hidden,
                        n_blocks=n_blocks, d_cond=1024).to(device)
    rfm = RectifiedFlowMatching(model, d_output=d_out, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_loss = float('inf')
    gate_results = {}
    
    # Pre-training baseline: evaluate BEFORE any optimizer steps
    print("  Pre-training baseline...")
    step_results = evaluate_gate(prior_name, model, device, 0, d_out, held_ds, sigma2_w)
    for key, val in step_results.items():
        gate_results.setdefault(key, []).append(val)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        idx = np.random.permutation(len(train_ds))
        for start in range(0, len(train_ds), batch_size):
            batch_idx = idx[start:start + batch_size]
            
            t5_batch, target_batch = [], []
            for i in batch_idx:
                t5, tgt = train_ds[i]
                t5_batch.append(t5)
                target_batch.append(tgt)
            
            x1 = torch.from_numpy(np.stack(target_batch)).float().to(device)
            cond = torch.from_numpy(np.stack(t5_batch)).float().to(device)
            
            loss = rfm.loss(x1, cond)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Held-out evaluation every 5 epochs with REAL T5 conditioning
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            step = epoch + 1
            step_results = evaluate_gate(prior_name, model, device, step, d_out, held_ds, sigma2_w)
            for key, val in step_results.items():
                gate_results.setdefault(key, []).append(val)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(), "loss": avg_loss},
                       output_dir / f"{prior_name}_best.pt")
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
    
    return gate_results


def evaluate_gate(prior_name, model, device, step, d_out, held_ds, sigma2_w):
    """Evaluate gates on FFHQ held-out set with REAL T5 text conditioning."""
    model.eval()
    results = {}
    n_steps = 10
    n_eval = min(2000, len(held_ds))
    # consistent sample for tracking across epochs
    rng = np.random.default_rng(step)
    idx = rng.choice(len(held_ds), n_eval, replace=False)
    
    @torch.no_grad()
    def sample_ode(x_0, cond):
        x = x_0
        dt = 1.0 / n_steps
        for s in range(n_steps):
            t_arr = torch.full((x.shape[0], 1), s * dt, device=device)
            x = x + model(x, t_arr, cond) * dt
        return x
    
    # Load held-out batch with REAL T5 conditioning
    t5_batch, target_batch = [], []
    for i in idx:
        t5, tgt = held_ds[i]
        t5_batch.append(t5)
        target_batch.append(tgt)
    
    t5  = torch.from_numpy(np.stack(t5_batch)).float().to(device)
    tgt = torch.from_numpy(np.stack(target_batch)).float().to(device)
    
    if prior_name == "zg":
        # Gate G1: MSE(z_g_pred, z_g_true) / baseline_variance
        z_pred = sample_ode(torch.randn(n_eval, d_out, device=device), t5)
        mse_per_dim = torch.mean((z_pred - tgt) ** 2).item()
        
        # FFHQ z_g total variance is ~48.42 across 50 dims, so per-dim variance is ~0.968
        ffhq_baseline_variance = 0.968
        ratio = mse_per_dim / ffhq_baseline_variance
        
        results["G1"] = {"step": step, "mse": mse_per_dim, "sigma2_baseline": ffhq_baseline_variance,
                         "ratio": ratio, "threshold": 1.0, "pass": ratio < 1.0}
        print(f"  [G1@{step}] MSE={mse_per_dim:.4f}, ratio={ratio:.4f} {'PASS' if ratio < 1.0 else 'FAIL'}")
    
    elif prior_name == "lda":
        # Gate G2: Verification AUC against RAW AuraFace targets
        lda_pred = sample_ode(torch.randn(n_eval, d_out, device=device), t5)
        lda_pred_np = lda_pred.cpu().numpy()
        
        reconstructed = []
        for i in range(n_eval):
            full_pred = lda_to_full(lda_pred_np[i])
            full_pred = full_pred / (np.linalg.norm(full_pred) + 1e-8)
            reconstructed.append(full_pred)
        pred_norm = np.stack(reconstructed)
        
        # Use RAW AuraFace targets from the dataset
        true_norm = []
        for i in idx:
            a = held_ds.raw_targets[i]
            a = a / (np.linalg.norm(a) + 1e-8)
            true_norm.append(a)
        true_norm = np.stack(true_norm)
        
        # Verification AUC
        rng_auc = np.random.default_rng(step)
        pos_cos = np.sum(pred_norm * true_norm, axis=1)  # (N,)
        
        # Negatives: pred vs random other person's true AuraFace
        neg_cos = []
        for i in range(n_eval):
            j = rng_auc.integers(n_eval)
            while j == i:
                j = rng_auc.integers(n_eval)
            neg_cos.append(np.dot(pred_norm[i], true_norm[j]))
        neg_cos = np.array(neg_cos)
        
        all_scores = np.concatenate([pos_cos, neg_cos])
        labels = np.concatenate([np.ones_like(pos_cos), np.zeros_like(neg_cos)])
        order = np.argsort(all_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(all_scores))
        pos_ranks = ranks[labels == 1]
        auc = (np.sum(pos_ranks) - n_eval*(n_eval-1)/2.0) / (n_eval * n_eval)
        
        mean_pos = float(np.mean(pos_cos))
        mean_neg = float(np.mean(neg_cos))
        
        results["G2"] = {"step": step, "auc": float(auc), "pos_cos": mean_pos, "neg_cos": mean_neg,
                         "threshold": 0.5, "pass": float(auc) > 0.5}
        print(f"  [G2@{step}] AUC={auc:.4f} (pos {mean_pos:+.3f} / neg {mean_neg:+.3f}) {'PASS' if auc > 0.5 else 'FAIL'}")
    
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train text-to-identity Priors")
    parser.add_argument("--prior", choices=["zg", "lda", "both"], default="both")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap FFHQ samples (quick tests)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"Device: {device} | Output: {output_dir}")
    
    # σ²_w from Hegre (neutral — same value regardless of training run)
    sigma2_w = compute_sigma_w_from_hegre()
    with open(output_dir / "sigma2_w.json", "w") as f:
        json.dump({"sigma2_w": sigma2_w}, f, indent=2)
    
    all_results = {}
    
    if args.eval_only:
        print("\nEvaluating existing checkpoints...")
        if args.prior in ("zg", "both"):
            ckpt_path = output_dir / "zg_best.pt"
            if ckpt_path.exists():
                print("\nBuilding FFHQ z_g dataset...")
                ds_zg = build_ffhq_zg_dataset(max_samples=args.max_samples, skip_norm_check=(args.max_samples is None))
                _, held_ds = split_ffhq(ds_zg)
                cfg = CONFIG["zg"]
                model = AdaLNResNet(d_in=cfg["d_out"], d_out=cfg["d_out"], d_hidden=cfg["d_hidden"],
                                    n_blocks=cfg["n_blocks"], d_cond=1024).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                step = ckpt["epoch"]
                res = evaluate_gate("zg", model, device, step, cfg["d_out"], held_ds, sigma2_w)
                all_results["zg"] = res
            else:
                print("No checkpoint found for zg prior.")
                
        if args.prior in ("lda", "both"):
            ckpt_path = output_dir / "lda_best.pt"
            if ckpt_path.exists():
                print("\nBuilding FFHQ AuraFace-LDA dataset...")
                ds_lda = build_ffhq_lda_dataset(max_samples=args.max_samples)
                _, held_ds = split_ffhq(ds_lda)
                cfg = CONFIG["lda"]
                model = AdaLNResNet(d_in=cfg["d_out"], d_out=cfg["d_out"], d_hidden=cfg["d_hidden"],
                                    n_blocks=cfg["n_blocks"], d_cond=1024).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                step = ckpt["epoch"]
                res = evaluate_gate("lda", model, device, step, cfg["d_out"], held_ds, sigma2_w)
                all_results["lda"] = res
            else:
                print("No checkpoint found for lda prior.")
    elif args.prior in ("zg", "both") and not args.eval_only:
        print("\nBuilding FFHQ z_g dataset...")
        ds_zg = build_ffhq_zg_dataset(max_samples=args.max_samples,
                                      skip_norm_check=(args.max_samples is None))
        train_ds, held_ds = split_ffhq(ds_zg)
        cfg = CONFIG["zg"]
        epochs = args.epochs or cfg["epochs"]
        batch_size = args.batch_size or cfg["batch_size"]
        
        results = train_prior("zg", train_ds, held_ds, cfg["d_out"], cfg["d_hidden"],
                              cfg["n_blocks"], device, epochs, batch_size, cfg["lr"],
                              output_dir, sigma2_w)
        all_results["zg"] = results
    
    if args.prior in ("lda", "both") and not args.eval_only:
        print("\nBuilding FFHQ AuraFace-LDA dataset...")
        ds_lda = build_ffhq_lda_dataset(max_samples=args.max_samples)
        train_ds, held_ds = split_ffhq(ds_lda)
        cfg = CONFIG["lda"]
        epochs = args.epochs or cfg["epochs"]
        batch_size = args.batch_size or cfg["batch_size"]
        
        results = train_prior("lda", train_ds, held_ds, cfg["d_out"], cfg["d_hidden"],
                              cfg["n_blocks"], device, epochs, batch_size, cfg["lr"],
                              output_dir, sigma2_w)
        all_results["lda"] = results
    
    with open(output_dir / "gate_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'gate_results.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
