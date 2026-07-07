#!/usr/bin/env python3
"""
Step 3 — resolve the frontalization confound.
Compare Sapiens2 keypoints frontalized TWO ways, holding keypoint source constant:
  (A) measured-3D-GPA  — uses real pointmap depth (what gave 0.734)
  (B) template-lift    — z_g's exact recipe: lift 2D via rotated canonical-template Z
If A≈B → method irrelevant, source/density drives the gain (2D already shows +0.077).
If A>B → Sapiens2's MEASURED depth adds identity signal a template cannot.
"""
import numpy as np, json

d=np.load('study_features.npz',allow_pickle=True)
personas=d['personas']; setids=d['setids']; s2kp=d['s2kp']  # (N,308,3) 2D+conf
k3=np.load('study_kp3d.npz')['kp3d']                          # (N,308,3) measured 3D
N=len(personas)

valid3=np.isfinite(k3).all(2); stable=valid3.mean(0)>0.90; sidx=np.where(stable)[0]
K3=k3[:,sidx,:]; K2=s2kp[:,sidx,:2]
img_ok=np.isfinite(K3).all((1,2))
print(f"Stable kp: {sidx.size}, images usable: {img_ok.sum()}/{N}")

def gpa(shapes,iters=5):
    X=shapes-shapes.mean(1,keepdims=True); X=X/(np.sqrt((X**2).sum((1,2),keepdims=True))+1e-9)
    ref=X[0].copy()
    for _ in range(iters):
        out=np.zeros_like(X)
        for i in range(len(X)):
            M=ref.T@X[i]; U,_,Vt=np.linalg.svd(M); R=U@Vt
            if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
            out[i]=X[i]@R.T
        ref=out.mean(0); ref/=(np.sqrt((ref**2).sum())+1e-9); X=out
    return X, ref.reshape(-1,shapes.shape[2])

def zscore(X): return (X-np.nanmean(X,0))/(np.nanstd(X,0)+1e-9)
def vauc(F,lab,st,n_pairs=40000,seed=0):
    rng=np.random.RandomState(seed); Fz=zscore(F); Fn=Fz/(np.linalg.norm(Fz,axis=1,keepdims=True)+1e-9)
    by={}
    for i in range(len(lab)): by.setdefault(lab[i],[]).append(i)
    pl=[p for p in by if len(by[p])>=2]; same=[];diff=[]
    for _ in range(n_pairs):
        p=pl[rng.randint(len(pl))];m=by[p];i,j=m[rng.randint(len(m))],m[rng.randint(len(m))];tr=0
        while st[i]==st[j] and tr<10: j=m[rng.randint(len(m))];tr+=1
        if i!=j: same.append(float(Fn[i]@Fn[j]))
        p2=pl[rng.randint(len(pl))]
        while p2==p: p2=pl[rng.randint(len(pl))]
        a=by[p][rng.randint(len(by[p]))];b=by[p2][rng.randint(len(by[p2]))];diff.append(float(Fn[a]@Fn[b]))
    same=np.array(same);diff=np.array(diff);allv=np.concatenate([same,diff]);r=allv.argsort().argsort()
    return (r[:len(same)].sum()-len(same)*(len(same)-1)/2)/(len(same)*len(diff))

lab=personas[img_ok]; st=setids[img_ok]
K3o=K3[img_ok]; K2o=K2[img_ok]

# (A) measured-3D-GPA
FA,_=gpa(K3o); FA=FA.reshape(len(K3o),-1)
aucA=np.mean([vauc(FA,lab,st,seed=s) for s in (0,1,2)])

# Build canonical template from GPA mean of measured 3D
_,template=gpa(K3o)   # (K,3) mean frontal shape

# (B) template-lift (z_g recipe) on 2D keypoints
def estimate_R_s(X, y):
    # align template XY (X[:,:2]) to observed 2D y via similarity (rotation+scale, 2D)
    Xc=X[:,:2]-X[:,:2].mean(0); yc=y-y.mean(0)
    H=Xc.T@yc; U,_,Vt=np.linalg.svd(H); R2=(Vt.T@U.T)
    if np.linalg.det(R2)<0: Vt[-1]*=-1; R2=Vt.T@U.T
    s=np.trace(yc.T@Xc@R2)/ (np.trace(Xc.T@Xc)+1e-9)
    return R2, s
lifted=np.zeros((len(K2o), sidx.size, 3))
for i in range(len(K2o)):
    R2,s=estimate_R_s(template, K2o[i])
    # rotated template Z scaled to observed space (z_g's pitfall-corrected recipe)
    camZ = (template[:,2] - template[:,2].mean())  # template depth
    lifted[i,:,:2]=K2o[i]
    lifted[i,:,2]=camZ * s
FB,_=gpa(lifted); FB=FB.reshape(len(K2o),-1)
aucB=np.mean([vauc(FB,lab,st,seed=s) for s in (0,1,2)])

print(f"\n=== STEP 3: frontalization method control (Sapiens2 keypoints, source held constant) ===")
print(f"  (A) measured-3D-GPA (real pointmap depth): {aucA:.4f}")
print(f"  (B) template-lift  (z_g recipe, template depth): {aucB:.4f}")
print(f"  Δ(A-B) = {aucA-aucB:+.4f}")
print(f"\n  2D shape (no depth, same-method source comparison): Sapiens2 0.766 vs DWPose 0.688 (+0.077)")
print(f"  documented z_g (DWPose-68, template-lift): 0.67-0.69")
print()
if aucA - aucB > 0.02:
    print("  → Sapiens2 MEASURED depth adds identity signal beyond a template (real geometry helps).")
elif aucB > 0.71:
    print("  → template-lift alone (like z_g) already clears 0.71 on Sapiens2 kp → DENSITY/SOURCE is the driver, not my GPA method.")
else:
    print("  → template-lift drops below gate; the measured-GPA number leaned on method. Report 2D source result (+0.077) as the clean claim.")
json.dump({'A_measured_gpa':float(aucA),'B_template_lift':float(aucB)},open('study_results_step3.json','w'),indent=2)
