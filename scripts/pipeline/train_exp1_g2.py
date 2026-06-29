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
MAX_TOKENS = 256    # cover full real captions (~150-210 valid tokens, padded to 512)


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
    """Load a batch from disk. Returns (t5, lda_target, raw_aura).

    Uses t5_mask to ignore padding tokens: mean-pool (Arm A) averages only real
    tokens; full-seq (Arm B/C) keeps real tokens up to MAX_TOKENS. Padding-token
    contamination would otherwise handicap both conditioning paths.
    """
    t5s, ldas, raws = [], [], []
    for fid in ids:
        raw = np.load(FFHQ / "auraface" / f"{fid}.npy").astype(np.float32).ravel()
        lda = project_to_lda(clean_auraface(raw.astype(np.float64))).ravel().astype(np.float32)
        t5 = np.load(FFHQ / "stratum" / fid / "t5_hidden.npy").astype(np.float32)
        mask_f = FFHQ / "stratum" / fid / "t5_mask.npy"
        if mask_f.exists():
            m = np.load(mask_f).astype(bool)
        else:
            m = np.ones(t5.shape[0], dtype=bool)
        valid = t5[m]                                  # (n_real, 1024)
        if pool_t5:
            t5 = valid.mean(axis=0) if len(valid) else t5.mean(axis=0)  # (1024,)
        else:
            seq = valid[:MAX_TOKENS]                    # (<=MAX_TOKENS, 1024)
            # pad to MAX_TOKENS so the batch stacks; cross-attn ignores zero rows
            if len(seq) < MAX_TOKENS:
                seq = np.vstack([seq, np.zeros((MAX_TOKENS - len(seq), seq.shape[1]), dtype=np.float32)])
            t5 = seq
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


# --- Attribute-consistency gate -------------------------------------------------
SKIN_TERMS = {
    "fair": ["fair skin", "pale skin", "light skin"],
    "light_brown": ["light brown skin", "tan skin", "olive skin"],
    "dark": ["dark skin", "brown skin", "deep skin"],
}
HAIR_TERMS = {
    "blonde": ["blonde hair", "blond hair"],
    "dark": ["dark hair", "black hair", "brown hair"],
    "red": ["red hair", "auburn hair"],
    "gray": ["gray hair", "grey hair", "white hair"],
}

def _attr_label(caption, term_map):
    c = caption.lower()
    for label, terms in term_map.items():
        if any(t in c for t in terms):
            return label
    return None

def attribute_consistency_auc(pred_lda, held_ids, raw_true, ffhq_root, seed=0):
    """Does the predicted identity match the DESCRIBED attribute better than chance?

    For each held-out sample with a parseable skin/hair attribute:
      positive = mean cosine of pred vs REAL faces sharing that attribute (excl self)
      negative = mean cosine of pred vs REAL faces NOT sharing it
    Returns AUC over (pos, neg) pairs. >0.5 means text attributes steer the
    prediction toward the right attribute cluster — distinct from pinning the
    exact person (verification AUC).
    """
    import numpy as np
    n = len(pred_lda)
    pred_norm = np.stack([
        (lambda f: f / (np.linalg.norm(f) + 1e-8))(lda_to_full(pred_lda[i]))
        for i in range(n)
    ])
    true_norm = raw_true / (np.linalg.norm(raw_true, axis=1, keepdims=True) + 1e-8)

    skin_labels, hair_labels = [], []
    for fid in held_ids:
        cf = Path(ffhq_root) / "stratum" / fid / "caption.txt"
        txt = cf.read_text() if cf.exists() else ""
        skin_labels.append(_attr_label(txt, SKIN_TERMS))
        hair_labels.append(_attr_label(txt, HAIR_TERMS))
    skin_labels = np.array(skin_labels, dtype=object)
    hair_labels = np.array(hair_labels, dtype=object)

    def _auc_for(labels):
        pos_scores, neg_scores = [], []
        for i in range(n):
            li = labels[i]
            if li is None:
                continue
            same = np.array([j for j in range(n) if j != i and labels[j] == li])
            diff = np.array([j for j in range(n) if labels[j] is not None and labels[j] != li])
            if len(same) < 3 or len(diff) < 3:
                continue
            pos_scores.append(np.mean(pred_norm[i] @ true_norm[same].T))
            neg_scores.append(np.mean(pred_norm[i] @ true_norm[diff].T))
        if len(pos_scores) < 10:
            return None, 0
        pos = np.array(pos_scores); neg = np.array(neg_scores)
        m = len(pos)
        alls = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(m), np.zeros(m)])
        order = np.argsort(alls); ranks = np.empty_like(order); ranks[order] = np.arange(len(alls))
        auc = (ranks[lab == 1].sum() - m*(m-1)/2) / (m*m)
        return float(auc), m

    skin_auc, n_skin = _auc_for(skin_labels)
    hair_auc, n_hair = _auc_for(hair_labels)
    return {"skin_auc": skin_auc, "n_skin": n_skin, "hair_auc": hair_auc, "n_hair": n_hair}


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

    # Final attribute-consistency gate (does the prediction match DESCRIBED traits?)
    model.eval()
    n_eval = min(2000, len(held_ids))
    ev_ids = held_ids[:n_eval]
    preds = []
    with torch.no_grad():
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
    _, _, raw = load_batch(ev_ids, pool_t5=True)
    attr = attribute_consistency_auc(pred_lda, ev_ids, raw, FFHQ)
    print(f"  [ATTR] skin_auc={attr['skin_auc']} (n={attr['n_skin']}) "
          f"hair_auc={attr['hair_auc']} (n={attr['n_hair']})")

    torch.save({"model_state": model.state_dict(), "arm": arm}, out_dir / f"exp1_arm_{arm}.pt")
    return {"history": history, "attribute": attr}


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

    print("\n=== SUMMARY (final verification AUC + attribute AUC) ===")
    for arm, res in results.items():
        h = res["history"][-1]; a = res["attribute"]
        print(f"  Arm {arm}: verif_AUC={h['auc']:.4f} | skin={a['skin_auc']} hair={a['hair_auc']}")
    if "A" in results and "B" in results:
        delta = results["B"]["history"][-1]["auc"] - results["A"]["history"][-1]["auc"]
        print(f"  B - A = {delta:+.4f} (credit hypothesis if >= +0.05)")
    print(f"\nResults: {out_dir / 'exp1_results.json'}")


if __name__ == "__main__":
    main()
