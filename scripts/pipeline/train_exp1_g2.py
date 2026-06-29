#!/usr/bin/env python3
"""
Experiment #1: G2 conditioning fix — three arms, corrected metrics.

Tests whether full-sequence T5 cross-attention beats mean-pooling for text->identity.

Arm A (baseline):  mean-pooled T5 (1024-d) -> FM AdaLNResNet
Arm B (full-seq):  full T5 sequence (S,1024) -> FM AdaLNResNetCrossAttn
Arm C (regressor): full T5 sequence -> deterministic IdentityRegressor (no FM)

Primary gate G2': Verification AUC of predicted identity vs RAW AuraFace on
held-out FFHQ tail. Credit the conditioning hypothesis only if B beats A by
>= +0.05 AUC.

Data is streamed from disk per-batch (full T5 sequences are too large to preload:
512x1024 fp16 = 1MB/sample, 60k = 63GB).
"""
import os, sys, argparse, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiments" / "geometry_pca"))

import torch
from priors.models_torch import AdaLNResNet, AdaLNResNetCrossAttn, IdentityRegressor
from priors.flow_matching_torch import RectifiedFlowMatching
from geometry_pca.auraface_preprocessing import clean_auraface, project_to_lda, lda_to_full

FFHQ = Path("/mnt/nas-ai-models/training-data/ffhq")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_OUT = 64          # AuraFace-LDA target dim
D_COND = 1024       # T5 hidden dim
MAX_TOKENS = 128    # cap T5 sequence length for memory/speed (identity content is early)


def build_index(max_samples=None):
    """List FFHQ ids that have both AuraFace and T5. Returns sorted list of ids."""
    aura_dir = FFHQ / "auraface"
    ids = []
    for af in sorted(os.listdir(aura_dir)):
        if not af.endswith(".npy"):
            continue
        fid = af[:-4]
        if (FFHQ / "stratum" / fid / "t5_hidden.npy").exists():
            ids.append(fid)
        if max_samples and len(ids) >= max_samples:
            break
    return ids


def load_batch(ids, pool_t5):
    """Load a batch from disk. Returns (t5, lda_target, raw_aura)."""
    t5s, ldas, raws = [], [], []
    for fid in ids:
        raw = np.load(FFHQ / "auraface" / f"{fid}.npy").astype(np.float32).ravel()
        lda = project_to_lda(clean_auraface(raw.astype(np.float64))).ravel().astype(np.float32)
        t5 = np.load(FFHQ / "stratum" / fid / "t5_hidden.npy").astype(np.float32)
        if pool_t5:
            t5 = t5.mean(axis=0)                  # (1024,)
        else:
            t5 = t5[:MAX_TOKENS]                  # (<=128, 1024)
        t5s.append(t5); ldas.append(lda); raws.append(raw)
    return np.stack(t5s), np.stack(ldas), np.stack(raws)


def verification_auc(pred_lda, raw_true, seed=0):
    """G2': reconstruct pred LDA -> 512d -> L2norm, verify vs RAW AuraFace via AUC."""
    n = len(pred_lda)
    pred_norm = np.stack([
        (lambda f: f / (np.linalg.norm(f) + 1e-8))(lda_to_full(pred_lda[i]))
        for i in range(n)
    ])
    true_norm = raw_true / (np.linalg.norm(raw_true, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)
    pos = np.sum(pred_norm * true_norm, axis=1)
    neg = np.array([np.dot(pred_norm[i], true_norm[(lambda j: j if j != i else (j+1) % n)(rng.integers(n))])
                    for i in range(n)])
    alls = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(n), np.zeros(n)])
    order = np.argsort(alls); ranks = np.empty_like(order); ranks[order] = np.arange(len(alls))
    auc = (ranks[lab == 1].sum() - n*(n-1)/2) / (n*n)
    return float(auc), float(pos.mean()), float(neg.mean())


def train_arm(arm, train_ids, held_ids, epochs, batch_size, lr, device, out_dir):
    pool_t5 = (arm == "A")
    is_fm = (arm in ("A", "B"))

    if arm == "A":
        model = AdaLNResNet(d_in=D_OUT, d_out=D_OUT, d_hidden=1024, n_blocks=12, d_cond=D_COND)
    elif arm == "B":
        model = AdaLNResNetCrossAttn(d_in=D_OUT, d_out=D_OUT, d_hidden=1024, n_blocks=12,
                                     d_cond=D_COND, n_heads=8, n_queries=4)
    else:  # C
        model = IdentityRegressor(d_out=D_OUT, d_hidden=1024, n_blocks=8,
                                  d_cond=D_COND, n_heads=8, n_queries=4)
    model = model.to(device)
    rfm = RectifiedFlowMatching(model, d_output=D_OUT, device=device) if is_fm else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n{'='*60}\n Arm {arm}: {'mean-pool' if pool_t5 else 'full-seq cross-attn'}"
          f" | {'FM' if is_fm else 'regressor'}\n"
          f" train={len(train_ids)} held={len(held_ids)} ep={epochs} bs={batch_size}\n{'='*60}")

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        n_eval = min(2000, len(held_ids))
        ev_ids = held_ids[:n_eval]
        preds = []
        for s in range(0, n_eval, batch_size):
            bids = ev_ids[s:s+batch_size]
            t5, _, _ = load_batch(bids, pool_t5)
            cond = torch.from_numpy(t5).float().to(device)
            if is_fm:
                x = torch.randn(len(bids), D_OUT, device=device); dt = 0.1
                for k in range(10):
                    t = torch.full((len(bids), 1), k*dt, device=device)
                    x = x + model(x, t, cond) * dt
                preds.append(x.cpu().numpy())
            else:
                preds.append(model(cond).cpu().numpy())
        pred_lda = np.concatenate(preds)
        _, _, raw = load_batch(ev_ids, pool_t5=True)  # raw aura same regardless of pooling
        auc, p, ng = verification_auc(pred_lda, raw)
        print(f"  [G2'@{step}] AUC={auc:.4f} (pos {p:+.3f}/neg {ng:+.3f})")
        model.train()
        return {"step": step, "auc": auc, "pos": p, "neg": ng}

    history = [evaluate(0)]
    for ep in range(epochs):
        idx = np.random.permutation(len(train_ids))
        ep_loss, nb = 0.0, 0
        for s in range(0, len(train_ids), batch_size):
            bids = [train_ids[i] for i in idx[s:s+batch_size]]
            t5, lda, _ = load_batch(bids, pool_t5)
            cond = torch.from_numpy(t5).float().to(device)
            tgt = torch.from_numpy(lda).float().to(device)
            if is_fm:
                loss = rfm.loss(tgt, cond)
            else:
                loss = torch.mean((model(cond) - tgt) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 5 == 0 or ep == epochs-1:
            history.append(evaluate(ep+1))
        print(f"  ep {ep+1}/{epochs} loss={ep_loss/max(nb,1):.4f}")

    torch.save({"model_state": model.state_dict(), "arm": arm}, out_dir / f"exp1_arm_{arm}.pt")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--held-fraction", type=float, default=0.15)
    ap.add_argument("--output", default="output/exp1_g2")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | building index...")
    ids = build_index(args.max_samples)
    n_held = max(1, int(len(ids) * args.held_fraction))
    train_ids, held_ids = ids[:-n_held], ids[-n_held:]
    print(f"  {len(ids)} usable FFHQ ids | train={len(train_ids)} held={len(held_ids)}")

    results = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        t0 = time.time()
        results[arm] = train_arm(arm, train_ids, held_ids, args.epochs,
                                 args.batch_size, args.lr, DEVICE, out_dir)
        print(f"  Arm {arm} done in {time.time()-t0:.0f}s")
        with open(out_dir / "exp1_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY (final AUC) ===")
    for arm, hist in results.items():
        print(f"  Arm {arm}: {hist[-1]['auc']:.4f}")
    if "A" in results and "B" in results:
        delta = results["B"][-1]["auc"] - results["A"][-1]["auc"]
        print(f"  B - A = {delta:+.4f} (credit hypothesis if >= +0.05)")
    print(f"\nResults: {out_dir / 'exp1_results.json'}")


if __name__ == "__main__":
    main()
