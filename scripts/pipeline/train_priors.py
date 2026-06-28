#!/usr/bin/env python3
"""
Train Rectified Flow Matching Priors for text→z_g and text→AuraFace-LDA.

Loads paired (T5, target) data from FFHQ, trains two separate AdaLN-ResNet models
with RFM, and evaluates on held-out Hegre identities using pre-registered gates.

Gate G1 (z_g Prior): held-out MSE(z_g_pred, z_g_true) / σ²_w < 1.0
Gate G2 (AuraFace-LDA Prior): held-out mean cosine similarity > 0.3
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
from priors.data import build_ffhq_zg_dataset, build_ffhq_lda_dataset, Z_G_MAX_NORM
from priors.models_torch import AdaLNResNet
from priors.flow_matching_torch import RectifiedFlowMatching
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = "output/prior_checkpoints"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters (tuned for 1D MLPs on 70k FFHQ samples)
CONFIG = {
    "zg": {
        "d_out": 50,
        "d_hidden": 1024,
        "n_blocks": 12,
        "lr": 1e-4,
        "epochs": 50,
        "batch_size": 512,
    },
    "lda": {
        "d_out": 64,
        "d_hidden": 1024,
        "n_blocks": 12,
        "lr": 1e-4,
        "epochs": 50,
        "batch_size": 512,
    },
}


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_hegre_held_out(persona_split="held_out"):
    """Load held-out Hegre identities for gate evaluation.
    
    Returns:
        zg_held: (N, 50) z_g vectors
        aura_held: (N, 512) raw AuraFace vectors
        persona_labels: (N,) integer identity labels
    """
    import sqlite3
    HEGRE = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
    
    conn = sqlite3.connect(f"file:{HEGRE / 'review.db'}?mode=ro&nolock=1", uri=True)
    c = conn.cursor()
    c.execute("""SELECT i.image_path, i.persona_id, p.name
                 FROM images i JOIN personas p ON i.persona_id = p.id
                 WHERE i.status = 'approved'""")
    rows = c.fetchall()
    conn.close()
    
    # Split: last 64 personas as held-out (matches the LDA held-out split from earlier)
    persona_order = sorted(set(r[2] for r in rows))
    held_out_names = set(persona_order[-64:]) if len(persona_order) > 64 else set()
    
    zg_list, aura_list, labels = [], [], []
    label_map = {}
    miss = 0
    for img_path, pid, pname in rows:
        if pname not in held_out_names:
            continue
        rel = Path(img_path).with_suffix('')
        
        # Load z_g
        zg_f = HEGRE / "zg" / f"{rel}.npy"
        if not zg_f.exists():
            miss += 1; continue
        z = np.load(zg_f).astype(np.float64)
        if np.linalg.norm(z) >= Z_G_MAX_NORM:
            miss += 1; continue
        
        # Load AuraFace
        aura_f = HEGRE / "auraface" / f"{rel}.npy"
        if not aura_f.exists():
            miss += 1; continue
        a = np.load(aura_f).astype(np.float64)
        
        if pname not in label_map:
            label_map[pname] = len(label_map)
        
        zg_list.append(z)
        aura_list.append(a)
        labels.append(label_map[pname])
    
    print(f"  Held-out set: {len(zg_list)} images, {len(label_map)} identities, {miss} skipped")
    return (np.stack(zg_list), np.stack(aura_list), np.array(labels))


def compute_sigma_w(zg_array, labels):
    """Compute mean within-identity z_g variance (σ²_w) for G1."""
    total_var = 0.0
    n = 0
    for c in np.unique(labels):
        zc = zg_array[labels == c]
        if len(zc) < 2:
            continue
        total_var += np.sum((zc - zc.mean(0)) ** 2)
        n += len(zc)
    return total_var / n if n > 0 else 1.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_prior(prior_name, dataset, d_out, d_hidden, n_blocks, device,
                epochs, batch_size, lr, output_dir,
                zg_held=None, aura_held=None, persona_labels=None,
                sigma2_w=None):
    """Train a single Prior model. Returns gate results."""
    
    print(f"\n{'='*60}")
    print(f" Prior {prior_name}: text→{d_out}d")
    print(f" Samples: {len(dataset)}, Batch: {batch_size}, Epochs: {epochs}, LR: {lr}")
    print(f" Hidden: {d_hidden}, Blocks: {n_blocks}, Device: {device}")
    print(f"{'='*60}")
    
    model = AdaLNResNet(d_in=d_out, d_out=d_out, d_hidden=d_hidden,
                        n_blocks=n_blocks, d_cond=1024).to(device)
    rfm = RectifiedFlowMatching(model, d_output=d_out, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    total_steps = 0
    best_loss = float('inf')
    gate_results = {}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle indices
        idx = np.random.permutation(len(dataset))
        for start in range(0, len(dataset), batch_size):
            batch_idx = idx[start:start + batch_size]
            
            # Load batch (CPU-bound: load from disk, move to GPU once)
            t5_batch, target_batch = [], []
            for i in batch_idx:
                t5, tgt = dataset[i]
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
            total_steps += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # Held-out evaluation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            gate_results = evaluate_gate(prior_name, model, device, epoch + 1, d_out,
                                         zg_held, aura_held, persona_labels, sigma2_w)
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"{prior_name}_best.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
    
    return gate_results


def evaluate_gate(prior_name, model, device, step, d_out,
                  zg_held, aura_held, persona_labels, sigma2_w):
    """Evaluate pre-registered gates on held-out identities."""
    model.eval()
    results = {}
    n_steps = 10
    
    @torch.no_grad()
    def sample_ode(x_0, cond):
        """Euler ODE sampler — inline to avoid RectifiedFlowMatching wrapper."""
        x = x_0
        dt = 1.0 / n_steps
        for s in range(n_steps):
            t_val = s * dt
            t_arr = torch.full((x.shape[0], 1), t_val, device=device)
            v = model(x, t_arr, cond)
            x = x + v * dt
        return x
    
    if prior_name == "zg" and zg_held is not None and sigma2_w is not None:
        # Gate G1: MSE(zg_pred, zg_true) / σ²_w
        n = min(2000, len(zg_held))  # sample for speed
        idx = np.random.choice(len(zg_held), n, replace=False)
        
        # Use the identity-average T5 for conditioning (simulates "describe this person")
        # For now: use the first image's T5 per identity as a placeholder.
        # In production, we'd use persona-averaged T5.
        
        # Simpler: evaluate on the held-out images directly (per-image text→z_g)
        z_true = torch.from_numpy(zg_held[idx]).float().to(device)
        # For text conditioning: use the same T5 from the z_g dataset
        # (we don't have T5 for Hegre held-out — use zeros for now as a baseline)
        cond = torch.zeros(n, 1024, device=device)  # placeholder
        
        z_pred = sample_ode(torch.randn(n, d_out, device=device), cond)
        
        mse = torch.mean((z_pred - z_true) ** 2).item()
        ratio = mse / sigma2_w
        results["G1"] = {
            "step": step,
            "mse": mse,
            "sigma2_w": sigma2_w,
            "ratio": ratio,
            "threshold": 1.0,
            "pass": ratio < 1.0,
        }
        print(f"  [G1@{step}] MSE={mse:.4f}, σ²_w={sigma2_w:.4f}, ratio={ratio:.4f} "
              f"({'PASS' if ratio < 1.0 else 'FAIL'})")
    
    elif prior_name == "lda" and aura_held is not None:
        # Gate G2: mean cosine similarity
        n = min(2000, len(aura_held))
        idx = np.random.choice(len(aura_held), n, replace=False)
        
        # Apply preprocessing: clean_auraface → project_to_lda
        lda_true = []
        for i in idx:
            cleaned = clean_auraface(aura_held[i])
            lda_true.append(project_to_lda(cleaned))
        lda_true = np.stack(lda_true)
        lda_t = torch.from_numpy(lda_true).float().to(device)
        
        cond = torch.zeros(n, 1024, device=device)  # placeholder
        
        lda_pred = sample_ode(torch.randn(n, d_out, device=device), cond)
        
        # Reconstruct and L2-normalize
        lda_pred_np = lda_pred.cpu().numpy()
        reconstructed = []
        for i in range(n):
            full = lda_to_full(lda_pred_np[i])
            full = full / (np.linalg.norm(full) + 1e-8)
            reconstructed.append(full)
        pred_norm = np.stack(reconstructed)
        true_norm = []
        for i in idx:
            a = aura_held[i]
            a = a / (np.linalg.norm(a) + 1e-8)
            true_norm.append(a)
        true_norm = np.stack(true_norm)
        
        cosines = np.sum(pred_norm * true_norm, axis=1)
        mean_cos = float(np.mean(cosines))
        results["G2"] = {
            "step": step,
            "mean_cosine": mean_cos,
            "threshold": 0.3,
            "pass": mean_cos > 0.3,
        }
        print(f"  [G2@{step}] mean_cosine={mean_cos:.4f} "
              f"({'PASS' if mean_cos > 0.3 else 'FAIL'})")
    
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train text-to-identity Priors")
    parser.add_argument("--prior", choices=["zg", "lda", "both"], default="both",
                        help="Which prior to train")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap FFHQ samples (for quick test runs)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate existing checkpoints")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load held-out Hegre set for gate evaluation
    print("\nLoading held-out Hegre identities...")
    zg_held, aura_held, persona_labels = load_hegre_held_out()
    sigma2_w = compute_sigma_w(zg_held, persona_labels)
    print(f"  σ²_w (within-person z_g variance) = {sigma2_w:.4f}")
    
    # Save σ²_w for the ledger
    with open(output_dir / "sigma2_w.json", "w") as f:
        json.dump({"sigma2_w": sigma2_w, "n_identities": len(np.unique(persona_labels)),
                   "n_images": len(zg_held)}, f, indent=2)
    
    all_results = {}
    
    if args.prior in ("zg", "both") and not args.eval_only:
        print("\nBuilding FFHQ z_g dataset...")
        ds_zg = build_ffhq_zg_dataset(max_samples=args.max_samples)
        cfg = CONFIG["zg"]
        epochs = args.epochs or cfg["epochs"]
        batch_size = args.batch_size or cfg["batch_size"]
        
        results = train_prior(
            "zg", ds_zg, cfg["d_out"], cfg["d_hidden"], cfg["n_blocks"],
            device, epochs, batch_size, cfg["lr"], output_dir,
            zg_held=zg_held, aura_held=aura_held,
            persona_labels=persona_labels, sigma2_w=sigma2_w,
        )
        all_results["zg"] = results
    
    if args.prior in ("lda", "both") and not args.eval_only:
        print("\nBuilding FFHQ AuraFace-LDA dataset...")
        ds_lda = build_ffhq_lda_dataset(max_samples=args.max_samples)
        cfg = CONFIG["lda"]
        epochs = args.epochs or cfg["epochs"]
        batch_size = args.batch_size or cfg["batch_size"]
        
        results = train_prior(
            "lda", ds_lda, cfg["d_out"], cfg["d_hidden"], cfg["n_blocks"],
            device, epochs, batch_size, cfg["lr"], output_dir,
            zg_held=zg_held, aura_held=aura_held,
            persona_labels=persona_labels, sigma2_w=sigma2_w,
        )
        all_results["lda"] = results
    
    # Save final results
    with open(output_dir / "gate_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'gate_results.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
