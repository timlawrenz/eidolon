#!/usr/bin/env python3
"""
Fisher-J axis split on Sapiens2 frontalized keypoint shape (100 personas).
Tests whether dense Sapiens2 geometry SPLITS into:
  - low-J transient axes (pose/expression) → orthogonal z_g-replacement candidate
  - high-J morphology axes (identity shape) → editable sliders
Plus the decisive orthogonality check: is the low-J block still AuraFace-orthogonal?
Comparison: same pipeline on DWPose (did its morphology axes ever exist?).
"""
import numpy as np

d=np.load('/tmp/eidolon_3d_spike/study_features_wide.npz',allow_pickle=True)
s2kp=d['s2kp']; dwkp=d['dwkp']; af=d['af']; personas=d['personas']; setids=d['setids']
uniq=sorted(set(personas.tolist())); pidx={p:i for i,p in enumerate(uniq)}
y=np.array([pidx[p] for p in personas])

conf=s2kp[:,:,2]; stable=(conf>0.3).mean(0)>0.95; sidx=np.where(stable)[0]

def procrustes(shapes,iters=3):
    X=shapes-shapes.mean(1,keepdims=True); X=X/(np.sqrt((X**2).sum((1,2),keepdims=True))+1e-9)
    ref=X[0].copy()
    for _ in range(iters):
        out=np.zeros_like(X)
        for i in range(len(X)):
            M=ref.T@X[i]; U,_,Vt=np.linalg.svd(M); R=U@Vt
            if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
            out[i]=X[i]@R.T
        ref=out.mean(0); ref/=(np.sqrt((ref**2).sum())+1e-9); X=out
    return X.reshape(len(X),-1)

def pca(X,k=50):
    Xc=X-X.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    return Xc@Vt[:k].T, Vt[:k], S[:k]

def fisher_j(Z,y):
    """per-axis J = between-person var / within-person var."""
    J=np.zeros(Z.shape[1])
    gm=Z.mean(0)
    for a in range(Z.shape[1]):
        within=[]; means=[]
        for p in np.unique(y):
            v=Z[y==p,a]; 
            if len(v)>=2: within.append(v.var()); means.append(v.mean())
        wv=np.mean(within); bv=np.var(means)
        J[a]=bv/(wv+1e-9)
    return J

def zscore(X): return (X-np.nanmean(X,0))/(np.nanstd(X,0)+1e-9)
def vauc(F,lab,st,n=20000,seed=0):
    rng=np.random.RandomState(seed); Fz=zscore(F); Fn=Fz/(np.linalg.norm(Fz,axis=1,keepdims=True)+1e-9)
    by={}
    for i in range(len(lab)): by.setdefault(lab[i],[]).append(i)
    pl=[p for p in by if len(by[p])>=2]; same=[];diff=[]
    for _ in range(n):
        p=pl[rng.randint(len(pl))];m=by[p];i,j=m[rng.randint(len(m))],m[rng.randint(len(m))];tr=0
        while st[i]==st[j] and tr<10: j=m[rng.randint(len(m))];tr+=1
        if i!=j: same.append(float(Fn[i]@Fn[j]))
        p2=pl[rng.randint(len(pl))]
        while p2==p: p2=pl[rng.randint(len(pl))]
        a=by[p][rng.randint(len(by[p]))];b=by[p2][rng.randint(len(by[p2]))];diff.append(float(Fn[a]@Fn[b]))
    same=np.array(same);diff=np.array(diff);allv=np.concatenate([same,diff]);r=allv.argsort().argsort()
    return (r[:len(same)].sum()-len(same)*(len(same)-1)/2)/(len(same)*len(diff))

def ridge_r2(X,Y,seed=0,alpha=1.0):
    """held-out R² predicting Y (AuraFace) from X (transient block). person-split."""
    rng=np.random.RandomState(seed); n=len(X); perm=rng.permutation(n); tr=perm[:int(.8*n)]; te=perm[int(.8*n):]
    Xtr,Xte=X[tr],X[te]; Ytr,Yte=Y[tr],Y[te]
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    W=np.linalg.solve(Xtr.T@Xtr+alpha*np.eye(Xtr.shape[1]), Xtr.T@Ytr)
    pred=Xte@W; ss_res=((Yte-pred)**2).sum(); ss_tot=((Yte-Yte.mean(0))**2).sum()
    return 1-ss_res/ss_tot

print("=== Sapiens2 frontalized shape → PCA(50) → Fisher-J split ===")
Xs=procrustes(s2kp[:,sidx,:2]); Zs,_,_=pca(Xs,50); Js=fisher_j(Zs,y)
order=np.argsort(-Js)
print(f"Per-axis Fisher J (sorted): top5={Js[order[:5]].round(3)}  bottom5={Js[order[-5:]].round(3)}")
morph=np.where(Js>0.15)[0]; trans=np.where(Js<0.05)[0]; mid=np.where((Js>=0.05)&(Js<=0.15))[0]
print(f"  morphology axes (J>0.15): {len(morph)}   transient (J<0.05): {len(trans)}   mid: {len(mid)}")
print(f"  global mean J: {Js.mean():.3f}")

af_ok=np.isfinite(af).all(1)
aucF=np.mean([vauc(Zs,personas,setids,seed=s) for s in(0,1,2)])
aucM=np.mean([vauc(Zs[:,morph],personas,setids,seed=s) for s in(0,1,2)]) if len(morph)>=2 else float('nan')
aucT=np.mean([vauc(Zs[:,trans],personas,setids,seed=s) for s in(0,1,2)]) if len(trans)>=2 else float('nan')
print(f"\n  Verification AUC — full 50-d: {aucF:.3f}")
print(f"  Verification AUC — morphology block ({len(morph)}-d): {aucM:.3f}  (want HIGH = carries identity)")
print(f"  Verification AUC — transient block ({len(trans)}-d): {aucT:.3f}   (want ~0.5 = identity-blind)")

# Orthogonality: does transient block predict AuraFace?
if len(trans)>=2:
    r2=np.mean([ridge_r2(Zs[af_ok][:,trans], af[af_ok], seed=s) for s in (0,1,2)])
    print(f"\n  ORTHOGONALITY: transient-block → AuraFace ridge R² = {r2:+.3f}  (want ~0 for clean z_g replacement)")
r2_full=np.mean([ridge_r2(Zs[af_ok], af[af_ok], seed=s) for s in (0,1,2)])
print(f"  full-shape → AuraFace ridge R² = {r2_full:+.3f}  (vs DWPose z_g's documented ~-0.03)")

# DWPose comparison
print("\n=== DWPose frontalized shape → PCA(50) → Fisher-J (did morphology axes ever exist?) ===")
Xd=procrustes(np.nan_to_num(dwkp[:,:,:2])); Zd,_,_=pca(Xd,50); Jd=fisher_j(Zd,y)
print(f"  DWPose: morphology axes (J>0.15): {(Jd>0.15).sum()}   transient (J<0.05): {(Jd<0.05).sum()}   global mean J: {Jd.mean():.3f}")
print(f"  Sapiens2 mean J {Js.mean():.3f} vs DWPose mean J {Jd.mean():.3f}  → {'Sapiens2 has MORE splittable morphology' if Js.mean()>Jd.mean() else 'similar'}")

import json
json.dump({'sapiens2_meanJ':float(Js.mean()),'n_morph':int(len(morph)),'n_trans':int(len(trans)),
           'auc_full':float(aucF),'auc_morph':float(aucM),'auc_trans':float(aucT),
           'ortho_transient_r2':float(r2) if len(trans)>=2 else None,'ortho_full_r2':float(r2_full),
           'dwpose_meanJ':float(Jd.mean())}, open('/tmp/eidolon_3d_spike/study_split.json','w'),indent=2)
print("\nSaved study_split.json")
