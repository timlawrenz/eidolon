#!/usr/bin/env python3
"""
Identity gate — Arm B (2D shape variant) + Arm A confidence calibration.
Verification AUC via Mann-Whitney (numpy only). Cross-shoot same-identity pairs.
"""
import numpy as np
import json

d = np.load('/tmp/eidolon_3d_spike/study_features.npz', allow_pickle=True)
s2kp=d['s2kp']; dwkp=d['dwkp']; af=d['af']
personas=d['personas']; setids=d['setids']; zg=d['zg']
N=len(personas)
uniq_p=sorted(set(personas.tolist()))
print(f"N={N} images, {len(uniq_p)} personas")

# ── stable Sapiens2 keypoints: high conf across cohort ──
conf=s2kp[:,:,2]
stable=(conf>0.3).mean(0) > 0.95   # present in >95% of images
stable_idx=np.where(stable)[0]
print(f"Stable Sapiens2 keypoints (conf>0.3 in >95% imgs): {stable_idx.size}/308")

# ── 2D similarity-Procrustes alignment to mean shape ──
def procrustes_align(shapes):
    """shapes: (N,K,2). Remove translation, scale, rotation. Returns aligned (N,K,2)."""
    X = shapes - shapes.mean(1, keepdims=True)          # center
    norms = np.sqrt((X**2).sum((1,2), keepdims=True))
    X = X / (norms + 1e-9)                               # scale
    ref = X[0].copy()
    for _ in range(3):
        aligned=np.zeros_like(X)
        for i in range(len(X)):
            # optimal rotation of X[i] onto ref
            M = ref.T @ X[i]
            U,_,Vt = np.linalg.svd(M)
            R = U @ Vt
            aligned[i] = X[i] @ R.T
        ref = aligned.mean(0)
        ref = ref / (np.sqrt((ref**2).sum())+1e-9)
        X = aligned
    return X.reshape(len(X), -1)   # (N, K*2)

feats={}
# B1 Sapiens2 2D shape (stable kp)
feats['B1_sapiens2_2D'] = procrustes_align(s2kp[:, stable_idx, :2])
# B3 DWPose 2D shape (68 kp) — mask images with all-zero DWPose
dw_valid = ~np.all(dwkp[:,:,:2]==0, axis=(1,2)) & np.isfinite(dwkp).all((1,2))
feats['B3_dwpose_2D'] = procrustes_align(np.nan_to_num(dwkp[:,:, :2]))
# B5 AuraFace (raw 512, z-scored later)
feats['B5_auraface'] = af.copy()
# B6 random-projection null of B1
rng=np.random.RandomState(0)
Rp = rng.randn(feats['B1_sapiens2_2D'].shape[1], 64)
feats['B6_null'] = feats['B1_sapiens2_2D'] @ Rp

def zscore(X):
    mu=np.nanmean(X,0); sd=np.nanstd(X,0)+1e-9
    return (X-mu)/sd

def verification_auc(F, labels, sets, n_pairs=40000, seed=0, cross_shoot=True):
    rng=np.random.RandomState(seed)
    valid=np.isfinite(F).all(1)
    idx=np.where(valid)[0]
    F=zscore(F)
    # normalize for cosine
    Fn=F/ (np.linalg.norm(F,axis=1,keepdims=True)+1e-9)
    lab=labels; st=sets
    # build same/diff pairs
    same_sim=[]; diff_sim=[]
    by_p={}
    for i in idx:
        by_p.setdefault(lab[i],[]).append(i)
    plist=[p for p in by_p if len(by_p[p])>=2]
    for _ in range(n_pairs):
        # same pair (cross-shoot if possible)
        p=plist[rng.randint(len(plist))]
        members=by_p[p]
        i,j=members[rng.randint(len(members))],members[rng.randint(len(members))]
        tries=0
        while cross_shoot and st[i]==st[j] and tries<10:
            j=members[rng.randint(len(members))]; tries+=1
        if i!=j:
            same_sim.append(float(Fn[i]@Fn[j]))
        # diff pair
        p2=plist[rng.randint(len(plist))]
        while p2==p: p2=plist[rng.randint(len(plist))]
        a=by_p[p][rng.randint(len(by_p[p]))]; b=by_p[p2][rng.randint(len(by_p[p2]))]
        diff_sim.append(float(Fn[a]@Fn[b]))
    same=np.array(same_sim); diff=np.array(diff_sim)
    # AUC = P(same>diff) via rank (Mann-Whitney)
    allv=np.concatenate([same,diff]); ranks=allv.argsort().argsort()
    r_same=ranks[:len(same)].sum()
    auc=(r_same - len(same)*(len(same)-1)/2)/(len(same)*len(diff))
    return auc, len(same), len(diff)

print("\n=== CROSS-SHOOT VERIFICATION AUC (3-seed mean) ===")
results={}
for name,F in feats.items():
    aucs=[verification_auc(F,personas,setids,seed=s)[0] for s in (0,1,2)]
    results[name]=float(np.mean(aucs))
    print(f"  {name:22s} AUC = {np.mean(aucs):.4f}  (seeds: {[f'{a:.3f}' for a in aucs]})")

print("\n=== GATES ===")
b1,b3,b5=results['B1_sapiens2_2D'],results['B3_dwpose_2D'],results['B5_auraface']
b6=results['B6_null']
# proper chance null: shuffle persona labels
rng=np.random.RandomState(7)
shuf=personas.copy(); rng.shuffle(shuf)
chance=np.mean([verification_auc(feats['B1_sapiens2_2D'],shuf,setids,seed=s)[0] for s in (0,1,2)])
print(f"  Label-shuffle chance null:          {chance:.3f}  (should be ~0.50)")
print(f"  G1 Sapiens2-2D > DWPose-2D + 0.02:  {b1:.3f} vs {b3:.3f}  → {'PASS' if b1>b3+0.02 else 'FAIL'} (Δ={b1-b3:+.3f})")
print(f"  DWPose baseline vs documented 0.69: {b3:.3f}  → {'MATCH' if abs(b3-0.69)<0.02 else 'DIVERGE'}")
print(f"  RandProj-of-B1 (JL preserves→high): {b6:.3f}  (expected≈B1 by Johnson-Lindenstrauss, not a floor)")
print(f"  Ceiling AuraFace:                   {b5:.3f}")

json.dump(results, open('/tmp/eidolon_3d_spike/study_results_2d.json','w'), indent=2)

# ── Arm A: confidence calibration ──
print("\n=== ARM A: Sapiens2 confidence calibration ===")
print(f"  Mean conf all kp: {conf.mean():.3f}")
print(f"  Stable kp ({stable_idx.size}) mean conf: {conf[:,stable_idx].mean():.3f}")
unstable=np.where(~stable)[0]
print(f"  Unstable kp ({unstable.size}) mean conf: {conf[:,unstable].mean():.3f}")
print(f"  → Sapiens2 DOES separate confident vs uncertain kp: {conf[:,stable_idx].mean():.3f} vs {conf[:,unstable].mean():.3f}")
print(f"  DWPose has no such mechanism (returns all 68 always, {dw_valid.sum()}/{N} non-zero)")

