#!/usr/bin/env python3
"""
FULL-CORPUS audit of z_g: does it encode POSE (yaw/pitch) and/or IDENTITY?
Settles the contradiction between:
  (A) settled docs: z_g "pose-invariant by construction", identity AUC 0.67-0.69
  (B) 2026-07-07 note (25-persona cohort): z_g->yaw R2=0.98, z_g->identity AUC=0.90

All CPU + NAS I/O. No GPU.
"""
import os, sys, glob, random, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.append("/home/tim/source/activity/eidolon/experiments/geometry_pca")
from geometry_pca.pose_normalize import estimate_rotation
from geometry_pca.gpa import center_and_scale
from geometry_pca.constants import FACE_SLICE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FFHQ_STRATUM = "/mnt/nas-ai-models/training-data/ffhq/stratum"
FFHQ_ZG      = "/mnt/nas-ai-models/training-data/ffhq/zg"
HEGRE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
ENC = "/home/tim/source/activity/eidolon/experiments/geometry_pca/output/encoder_production.npz"

enc = dict(np.load(ENC))
TPL = enc["canonical_template"].copy(); TPL[:,1] *= -1   # mirror encode_zg
random.seed(0); np.random.seed(0)

def euler_from_R(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = np.arctan2(R[2,1], R[2,2])
    yaw   = np.arctan2(-R[2,0], sy)
    roll  = np.arctan2(R[1,0], R[0,0])
    return np.degrees([yaw, pitch, roll])

def face_from_pose(pose):
    if pose.shape == (133,3): f = pose[FACE_SLICE,:2]
    elif pose.shape == (68,2): f = pose
    elif pose.shape == (68,3): f = pose[:,:2]
    else: return None
    if (f==0).all(): return None
    return f.astype(np.float32)

def angles_and_rawshape(pose):
    f = face_from_pose(pose)
    if f is None: return None
    try:
        R,s = estimate_rotation(TPL - TPL.mean(0), f - f.mean(0))
    except Exception:
        return None
    ang = euler_from_R(R)
    raw = center_and_scale(f).reshape(-1)   # (136,) pose-normalized-for-scale but NOT frontalized
    return ang, raw

# ---------------- FFHQ pose sample ----------------
def load_ffhq(sid):
    try:
        zg = np.load(f"{FFHQ_ZG}/{sid}/zg.npy")
        pose = np.load(f"{FFHQ_STRATUM}/{sid}/pose.npy")
    except Exception:
        return None
    ar = angles_and_rawshape(pose)
    if ar is None or np.linalg.norm(zg) > 25 or np.linalg.norm(zg)<1e-3: return None
    return zg.astype(np.float32), ar[0].astype(np.float32), ar[1].astype(np.float32)

t0=time.time()
ffhq_ids = sorted(os.listdir(FFHQ_ZG))
random.shuffle(ffhq_ids); ffhq_ids = ffhq_ids[:8000]
with ThreadPoolExecutor(64) as ex:
    ffhq = [r for r in ex.map(load_ffhq, ffhq_ids) if r is not None]
Zf = np.stack([r[0] for r in ffhq]); Af = np.stack([r[1] for r in ffhq]); Rf = np.stack([r[2] for r in ffhq])
print(f"[FFHQ] loaded {len(ffhq)} usable  ({time.time()-t0:.0f}s)")
print(f"[FFHQ] yaw deg: std={Af[:,0].std():.1f} p5/p50/p95={np.percentile(Af[:,0],[5,50,95]).round(1)}  |yaw|>15deg frac={np.mean(np.abs(Af[:,0])>15):.2f}")
print(f"[FFHQ] pitch deg: std={Af[:,1].std():.1f} p5/95={np.percentile(Af[:,1],[5,95]).round(1)}")

def ridge_r2(X, y, n_splits=5):
    idx=np.arange(len(X)); r2s=[]
    for s in range(n_splits):
        rng=np.random.RandomState(s); rng.shuffle(idx)
        cut=int(0.8*len(idx)); tr,te=idx[:cut],idx[cut:]
        sc=StandardScaler().fit(X[tr])
        m=Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        p=m.predict(sc.transform(X[te]))
        ss_res=((y[te]-p)**2).sum(); ss_tot=((y[te]-y[te].mean())**2).sum()
        r2s.append(1-ss_res/ss_tot)
    return float(np.mean(r2s))

print("\n=== FFHQ: how much POSE is in z_g vs in raw landmarks ===")
for j,name in [(0,'yaw'),(1,'pitch')]:
    r2_raw = ridge_r2(Af_raw:=Rf, y:=Af[:,j])   # raw 2D shape -> angle (ceiling)
    r2_zg  = ridge_r2(Zf, Af[:,j])              # z_g -> angle (survives frontalization?)
    print(f"  {name:5s}: raw-2D-shape R2={r2_raw:+.3f}   z_g R2={r2_zg:+.3f}   (retained {r2_zg/max(r2_raw,1e-6)*100:4.0f}% of raw pose signal)")

# ---------------- HEGRE full-persona sample ----------------
print("\n[HEGRE] building manifest across all personas...")
zgroot=f"{HEGRE}/zg/faces"; personas=sorted(os.listdir(zgroot))
manifest=[]  # (zg_path, persona, setid)
for p in personas:
    pdir=os.path.join(zgroot,p)
    if not os.path.isdir(pdir): continue
    shoots=[s for s in os.listdir(pdir) if os.path.isdir(os.path.join(pdir,s))]
    for si,sh in enumerate(shoots):
        shp=os.path.join(pdir,sh)
        for f in os.listdir(shp):
            if f.endswith('.npy'):
                manifest.append((os.path.join(shp,f), p, f"{p}::{sh}"))
print(f"[HEGRE] manifest {len(manifest)} images, {len(personas)} personas")

# sample up to K per persona spanning shoots
from collections import defaultdict
byp=defaultdict(list)
for m in manifest: byp[m[1]].append(m)
K=40; sample=[]
for p,items in byp.items():
    random.shuffle(items); sample += items[:K]
print(f"[HEGRE] sampling {len(sample)} images ({K}/persona cap) across {len(byp)} personas")

def load_h(m):
    zp,per,setid=m
    try: zg=np.load(zp)
    except Exception: return None
    if np.linalg.norm(zg)>25 or np.linalg.norm(zg)<1e-3: return None
    ap=zp.replace('/zg/','/auraface/')
    af=None
    try:
        if os.path.exists(ap): af=np.load(ap).astype(np.float32)
    except Exception: af=None
    # pose (nested)
    pp=zp.replace('/zg/','/stratum/').replace('.npy','/pose.npy')
    ang=None
    try:
        if os.path.exists(pp):
            ar=angles_and_rawshape(np.load(pp))
            if ar is not None: ang=ar[0].astype(np.float32)
    except Exception: ang=None
    return zg.astype(np.float32), per, setid, af, ang

t1=time.time()
with ThreadPoolExecutor(64) as ex:
    H=[r for r in ex.map(load_h, sample) if r is not None]
print(f"[HEGRE] loaded {len(H)} usable ({time.time()-t1:.0f}s)")
Zh=np.stack([r[0] for r in H]); per=np.array([r[1] for r in H]); setid=np.array([r[2] for r in H])
has_af=np.array([r[3] is not None for r in H]); has_ang=np.array([r[4] is not None for r in H])
print(f"[HEGRE] personas covered={len(set(per))}  with_auraface={has_af.sum()} with_pose={has_ang.sum()}")

# pose distribution + z_g->yaw on hegre
if has_ang.sum()>500:
    Ah=np.stack([r[4] for r in H if r[4] is not None]); Zh_ang=Zh[has_ang]
    print(f"[HEGRE] yaw deg: std={Ah[:,0].std():.1f} p5/50/95={np.percentile(Ah[:,0],[5,50,95]).round(1)}  |yaw|>15 frac={np.mean(np.abs(Ah[:,0])>15):.2f}")
    print(f"[HEGRE] z_g->yaw R2={ridge_r2(Zh_ang,Ah[:,0]):+.3f}  z_g->pitch R2={ridge_r2(Zh_ang,Ah[:,1]):+.3f}")

# ---------------- identity verification (cross-shoot, full personas) ----------------
def zscore(X): 
    mu=X.mean(0); sd=X.std(0)+1e-9; return (X-mu)/sd
def verify_auc(F, labels, sets, n_pairs=40000, seed=0, cross_shoot=True):
    rng=np.random.RandomState(seed)
    Fn=zscore(F); Fn=Fn/(np.linalg.norm(Fn,axis=1,keepdims=True)+1e-9)
    byp=defaultdict(list)
    for i,l in enumerate(labels): byp[l].append(i)
    plist=[p for p in byp if len(byp[p])>=2]
    same=[]; diff=[]
    for _ in range(n_pairs):
        p=plist[rng.randint(len(plist))]; mem=byp[p]
        i,j=mem[rng.randint(len(mem))],mem[rng.randint(len(mem))]
        tr=0
        while cross_shoot and sets[i]==sets[j] and tr<12:
            j=mem[rng.randint(len(mem))]; tr+=1
        if i!=j: same.append(float(Fn[i]@Fn[j]))
        p2=p
        while p2==p: p2=plist[rng.randint(len(plist))]
        a=byp[p][rng.randint(len(byp[p]))]; b=byp[p2][rng.randint(len(byp[p2]))]
        diff.append(float(Fn[a]@Fn[b]))
    same=np.array(same); diff=np.array(diff)
    allv=np.concatenate([same,diff]); ranks=allv.argsort().argsort()
    r=ranks[:len(same)].sum()
    return (r-len(same)*(len(same)-1)/2)/(len(same)*len(diff))

print("\n=== HEGRE identity verification (FULL personas, cross-shoot, 3-seed) ===")
auc_zg=[verify_auc(Zh,per,setid,seed=s) for s in (0,1,2)]
print(f"  z_g       AUC={np.mean(auc_zg):.4f}  seeds={[round(a,3) for a in auc_zg]}   (doc says 0.67-0.69; 25p-note said 0.90)")
# label-shuffle chance
sh=per.copy(); np.random.RandomState(7).shuffle(sh)
print(f"  z_g SHUF  AUC={np.mean([verify_auc(Zh,sh,setid,seed=s) for s in (0,1,2)]):.4f}  (chance ~0.50)")
if has_af.sum()>500:
    Zaf=Zh[has_af]; Afh=np.stack([r[3] for r in H if r[3] is not None]); peraf=per[has_af]; setaf=setid[has_af]
    auc_af=[verify_auc(Afh,peraf,setaf,seed=s) for s in (0,1,2)]
    print(f"  AuraFace  AUC={np.mean(auc_af):.4f}  seeds={[round(a,3) for a in auc_af]}   (identity ceiling)")
    # z_g -> AuraFace linear overlap (person-split ridge, multi-output R2)
    uids=sorted(set(peraf.tolist())); rng=np.random.RandomState(0); rng.shuffle(uids)
    cut=int(0.8*len(uids)); trp=set(uids[:cut])
    tr=np.array([p in trp for p in peraf]); te=~tr
    sc=StandardScaler().fit(Zaf[tr]); m=Ridge(alpha=1.0).fit(sc.transform(Zaf[tr]),Afh[tr])
    P=m.predict(sc.transform(Zaf[te]))
    ss_res=((Afh[te]-P)**2).sum(); ss_tot=((Afh[te]-Afh[tr].mean(0))**2).sum()
    print(f"  z_g->AuraFace person-split ridge R2={1-ss_res/ss_tot:+.3f}  (doc says ~-0.03 = zero linear overlap)")
print("\nDONE.")
