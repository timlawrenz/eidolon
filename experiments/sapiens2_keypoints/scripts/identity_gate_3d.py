#!/usr/bin/env python3
"""
Identity gate — Arm B G2: 3D-frontalized (GPA) shape.
Removes 3D rotation/translation/scale (out-of-plane pose) → isolates pure facial shape.
This is the decisive test: does dense pose-normalized landmark SHAPE carry identity
beyond the 68-pt z_g result (Fisher J≈0.06)?
"""
import numpy as np, json

d=np.load('/tmp/eidolon_3d_spike/study_features.npz',allow_pickle=True)
personas=d['personas']; setids=d['setids']; s2kp=d['s2kp']
k3=np.load('/tmp/eidolon_3d_spike/study_kp3d.npz')['kp3d']  # (N,308,3)
N=len(personas)

# stable keypoints present in 3D across >90% of images
valid3=np.isfinite(k3).all(2)
stable=valid3.mean(0)>0.90
sidx=np.where(stable)[0]
print(f"Stable 3D keypoints (>90% imgs): {sidx.size}/308")

# keep only images where all stable kp are valid
K3=k3[:,sidx,:]
img_ok=np.isfinite(K3).all((1,2))
print(f"Images with all stable 3D kp: {img_ok.sum()}/{N}")

def gpa_3d(shapes, iters=5):
    """Generalized Procrustes: remove translation, scale, 3D rotation. shapes (N,K,3)."""
    X=shapes-shapes.mean(1,keepdims=True)
    X=X/(np.sqrt((X**2).sum((1,2),keepdims=True))+1e-9)
    ref=X[0].copy()
    for _ in range(iters):
        out=np.zeros_like(X)
        for i in range(len(X)):
            M=ref.T@X[i]; U,_,Vt=np.linalg.svd(M); R=U@Vt
            if np.linalg.det(R)<0:
                U[:,-1]*=-1; R=U@Vt
            out[i]=X[i]@R.T
        ref=out.mean(0); ref/= (np.sqrt((ref**2).sum())+1e-9); X=out
    return X.reshape(len(X),-1)

def zscore(X): return (X-np.nanmean(X,0))/(np.nanstd(X,0)+1e-9)

def verification_auc(F,labels,sets,idx_ok,n_pairs=40000,seed=0):
    rng=np.random.RandomState(seed)
    Fz=zscore(F); Fn=Fz/(np.linalg.norm(Fz,axis=1,keepdims=True)+1e-9)
    by_p={}
    for i in np.where(idx_ok)[0]:
        by_p.setdefault(labels[i],[]).append(i)
    plist=[p for p in by_p if len(by_p[p])>=2]
    same=[];diff=[]
    for _ in range(n_pairs):
        p=plist[rng.randint(len(plist))];mem=by_p[p]
        i,j=mem[rng.randint(len(mem))],mem[rng.randint(len(mem))]
        tr=0
        while sets[i]==sets[j] and tr<10: j=mem[rng.randint(len(mem))];tr+=1
        if i!=j: same.append(float(Fn[i]@Fn[j]))
        p2=plist[rng.randint(len(plist))]
        while p2==p: p2=plist[rng.randint(len(plist))]
        a=by_p[p][rng.randint(len(by_p[p]))];b=by_p[p2][rng.randint(len(by_p[p2]))]
        diff.append(float(Fn[a]@Fn[b]))
    same=np.array(same);diff=np.array(diff)
    allv=np.concatenate([same,diff]);ranks=allv.argsort().argsort()
    return (ranks[:len(same)].sum()-len(same)*(len(same)-1)/2)/(len(same)*len(diff))

# B2: Sapiens2 3D-frontalized shape
F_b2=gpa_3d(K3[img_ok])
lab=personas[img_ok]; st=setids[img_ok]
ok=np.ones(img_ok.sum(),bool)
auc_b2=np.mean([verification_auc(F_b2,lab,st,ok,seed=s) for s in (0,1,2)])

# chance
rng=np.random.RandomState(7); shuf=lab.copy(); rng.shuffle(shuf)
chance=np.mean([verification_auc(F_b2,shuf,st,ok,seed=s) for s in (0,1,2)])

print(f"\n=== G2: 3D-FRONTALIZED SHAPE (pose removed via GPA) ===")
print(f"  B2 Sapiens2 3D-frontalized shape:  {auc_b2:.4f}")
print(f"  Label-shuffle chance:              {chance:.4f}")
print(f"  vs 2D shape (had pose): Sapiens2 0.766, DWPose 0.688")
print(f"  vs documented z_g (68pt 3D-front): 0.67-0.69")
print(f"\n  G2 (B2 > 0.71): {'PASS' if auc_b2>0.71 else 'FAIL'}")
if auc_b2>0.71:
    print("  → dense pose-normalized landmark SHAPE carries identity the 68-pt z_g missed")
else:
    print("  → confirms z_g: pose-removed landmark shape is identity-poor; identity is in texture/appearance not sparse geometry")
json.dump({'B2_sapiens2_3D_frontalized':float(auc_b2),'chance':float(chance)},
          open('/tmp/eidolon_3d_spike/study_results_3d.json','w'),indent=2)
