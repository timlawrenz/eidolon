#!/usr/bin/env python3
"""
Phase 4: Masked Patch Tokens (Semantic Face Isolation)

Tests whether Masked Average Pooling of DINOv3 patch tokens (filtered by Sapiens
segmentation) isolates biological identity and reduces shoot-level leakage better
than the global CLS token.

Reads: dinov3_cls, dinov3_patches, seg, pose (for confidence gating), metadata
       from hegre_faces_stratum.
Outputs: data/phase4_patch_pooling.json
"""
import os, sys, json, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc
from geometry_pca.fisher import restandardize

FACE = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
SEEDS = [0, 1, 2]
CONF = 0.5
FG_MIN = 0.30

# Sapiens classes
FLESH_CLASSES = {2, 23, 24, 25, 26}  # face_neck, upper_lip, lower_lip, teeth, tongue
HAIR_CLASSES = {3}

def block_pool_mask(mask, block_size=16):
    """Average-pools a 2D mask by block_size."""
    h, w = mask.shape
    gh, gw = h // block_size, w // block_size
    # Trim to exact multiples just in case, though stratum crops are divisible
    mask = mask[:gh*block_size, :gw*block_size]
    # Reshape and mean
    return mask.reshape(gh, block_size, gw, block_size).mean(axis=(1, 3))

def main():
    db = sqlite3.connect("file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro", uri=True)
    rows = db.execute("""
        SELECT i.enriched_dir, i.persona_id, i.set_id
        FROM images i
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (SELECT persona_id FROM images WHERE status = 'tainted:contamination')
        ORDER BY i.persona_id, i.id
    """).fetchall()
    db.close()

    data = {
        "cls": [],
        "patch_unmasked": [],
        "patch_flesh": [],
        "patch_flesh_hair": []
    }
    y_id, y_set = [], []
    n_missing, n_conf, n_seg, n_empty = 0, 0, 0, 0

    for ed_rel, pid, set_id in rows:
        leaf = ed_rel.split("hegre_enriched/", 1)[1]
        ed = os.path.join(FACE, leaf)
        try:
            pose = np.load(os.path.join(ed, "pose.npy")).astype(np.float32)
            seg = np.load(os.path.join(ed, "seg.npy"))
            cls_t = np.load(os.path.join(ed, "dinov3_cls.npy")).astype(np.float32)
            patch_t = np.load(os.path.join(ed, "dinov3_patches.npy")).astype(np.float32)
        except Exception:
            n_missing += 1
            continue

        if pose[23:91, 2].mean() < CONF:
            n_conf += 1
            continue
            
        if (seg > 0).mean() < FG_MIN:
            n_seg += 1
            continue

        # Build binary masks
        mask_flesh = np.isin(seg, list(FLESH_CLASSES)).astype(np.float32)
        mask_flesh_hair = np.isin(seg, list(FLESH_CLASSES | HAIR_CLASSES)).astype(np.float32)

        # Pool to patch grid
        grid_flesh = block_pool_mask(mask_flesh, 16)
        grid_flesh_hair = block_pool_mask(mask_flesh_hair, 16)

        # Flatten grid masks to match patch_t shape (N_patches, 1024)
        flat_flesh = grid_flesh.ravel()
        flat_flesh_hair = grid_flesh_hair.ravel()

        if len(flat_flesh) != patch_t.shape[0]:
            # Should never happen based on audit, but safety first
            continue

        # Threshold at 0.5
        idx_flesh = flat_flesh > 0.5
        idx_flesh_hair = flat_flesh_hair > 0.5

        if not idx_flesh.any() or not idx_flesh_hair.any():
            n_empty += 1
            continue

        # Compute means
        data["cls"].append(cls_t.ravel())
        data["patch_unmasked"].append(patch_t.mean(axis=0))
        data["patch_flesh"].append(patch_t[idx_flesh].mean(axis=0))
        data["patch_flesh_hair"].append(patch_t[idx_flesh_hair].mean(axis=0))
        
        y_id.append(pid)
        y_set.append(set_id)

    y_id = np.array(y_id, dtype=np.int32)
    y_set = np.array(y_set, dtype=np.int32)
    for k in data:
        data[k] = np.stack(data[k])

    print(f"Extracted {len(y_id)} images, {len(np.unique(y_id))} identities")
    print(f"Skipped: missing={n_missing} conf<{CONF}={n_conf} fg<{FG_MIN}={n_seg} empty-mask={n_empty}")
    print("\n" + "="*60)
    print("  PHASE 4: MASKED PATCH TOKEN IDENTITY & LEAKAGE")
    print("="*60)

    # Similarity helper for shoot leakage
    def mean_sim(Z):
        Zs = restandardize(Z)
        Zn = Zs / np.maximum(np.linalg.norm(Zs, axis=1, keepdims=True), 1e-8)
        rngp = np.random.default_rng(3)
        by_id = {}
        for i, lab in enumerate(y_id):
            by_id.setdefault(int(lab), []).append(i)
        same_set, cross_set = [], []
        for lab, idxs in by_id.items():
            if len(idxs) < 2: continue
            for _ in range(min(60, len(idxs) * 3)):
                a, b = rngp.choice(idxs, size=2, replace=False)
                s = float(Zn[a] @ Zn[b])
                if y_set[a] == y_set[b]: same_set.append(s)
                else: cross_set.append(s)
        ss = np.array(same_set); cs = np.array(cross_set)
        return float(ss.mean()), float(cs.mean()), float(ss.mean() - cs.mean())

    results = {}
    baseline_auc = None

    for name in ["cls", "patch_unmasked", "patch_flesh", "patch_flesh_hair"]:
        Z = data[name]
        auc = float(np.mean([verification_auc(Z, y_id, seed=s)[0] for s in SEEDS]))
        ss, cs, gap = mean_sim(Z)
        
        print(f"\n[{name}]")
        print(f"  AUC       : {auc:.4f}")
        print(f"  Shoot Gap : {gap:+.4f}  (Same: {ss:+.3f}  Cross: {cs:+.3f})")
        
        if name == "cls":
            baseline_auc = auc
            verdict = "BASELINE"
        else:
            if auc > baseline_auc and name.startswith("patch_flesh"):
                verdict = "PASS"
            else:
                verdict = "FAIL"
            print(f"  Verdict   : {verdict}")

        results[name] = {"auc": auc, "gap": gap, "same_shoot": ss, "cross_shoot": cs, "verdict": verdict}

    with open("data/phase4_patch_pooling.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
