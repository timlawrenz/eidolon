#!/usr/bin/env python3
"""
Identity verification of z_g on the CANONICAL Postgres approved set (project moved
off review.db -> PostgreSQL). Cross-shoot, full 324 personas. Also z_g->AuraFace
linear overlap. Pose findings from zg_full_corpus_audit.py are label-independent
and unaffected; this only re-checks the IDENTITY claim on source-of-truth labels.
"""
import os, sys, random, time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HEGRE="/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
random.seed(0); np.random.seed(0)

c=psycopg2.connect(host="192.168.86.49",dbname="eidolon",user="tim",password="c5mpk9")
cur=c.cursor()
cur.execute("""SELECT p.name, i.set_id, i.image_path
               FROM images i JOIN personas p ON i.persona_id=p.id
               WHERE i.status='approved'""")
rows=cur.fetchall(); c.close()
print(f"approved rows: {len(rows)}  personas: {len({r[0] for r in rows})}")

byp=defaultdict(list)
for name,setid,ipath in rows:
    byp[name].append((setid,ipath))
K=40; sample=[]
for name,items in byp.items():
    random.shuffle(items); sample+=[(name,)+it for it in items[:K]]
print(f"sampling {len(sample)} imgs ({K}/persona) across {len(byp)} personas")

def zpath(ipath): return f"{HEGRE}/zg/{os.path.splitext(ipath)[0]}.npy"
def apath(ipath): return f"{HEGRE}/auraface/{os.path.splitext(ipath)[0]}.npy"

def load(rec):
    name,setid,ipath=rec
    zp=zpath(ipath)
    try: zg=np.load(zp)
    except Exception: return None
    n=np.linalg.norm(zg)
    if n>25 or n<1e-3: return None
    af=None; ap=apath(ipath)
    try:
        if os.path.exists(ap): af=np.load(ap).astype(np.float32)
    except Exception: af=None
    return zg.astype(np.float32), name, int(setid), af

t=time.time()
with ThreadPoolExecutor(64) as ex:
    R=[r for r in ex.map(load, sample) if r is not None]
print(f"loaded {len(R)} usable z_g ({time.time()-t:.0f}s)")
Z=np.stack([r[0] for r in R]); per=np.array([r[1] for r in R]); st=np.array([r[2] for r in R])
has=np.array([r[3] is not None for r in R])
print(f"personas covered: {len(set(per))}  with_auraface: {has.sum()}")

def zscore(X): mu=X.mean(0); sd=X.std(0)+1e-9; return (X-mu)/sd
def verify(F,lab,sets,n_pairs=40000,seed=0,cross_shoot=True):
    rng=np.random.RandomState(seed)
    Fn=zscore(F); Fn=Fn/(np.linalg.norm(Fn,axis=1,keepdims=True)+1e-9)
    bp=defaultdict(list)
    for i,l in enumerate(lab): bp[l].append(i)
    pl=[p for p in bp if len(bp[p])>=2]
    same=[];diff=[]
    for _ in range(n_pairs):
        p=pl[rng.randint(len(pl))]; m=bp[p]
        i,j=m[rng.randint(len(m))],m[rng.randint(len(m))]; tr=0
        while cross_shoot and sets[i]==sets[j] and tr<12: j=m[rng.randint(len(m))]; tr+=1
        if i!=j: same.append(float(Fn[i]@Fn[j]))
        p2=p
        while p2==p: p2=pl[rng.randint(len(pl))]
        a=bp[p][rng.randint(len(bp[p]))]; b=bp[p2][rng.randint(len(bp[p2]))]
        diff.append(float(Fn[a]@Fn[b]))
    same=np.array(same);diff=np.array(diff)
    allv=np.concatenate([same,diff]); rk=allv.argsort().argsort()
    return (rk[:len(same)].sum()-len(same)*(len(same)-1)/2)/(len(same)*len(diff))

print("\n=== z_g identity verification — Postgres approved set, cross-shoot, 3-seed ===")
azg=[verify(Z,per,st,seed=s) for s in (0,1,2)]
print(f"  z_g       AUC={np.mean(azg):.4f}  seeds={[round(a,3) for a in azg]}")
sh=per.copy(); np.random.RandomState(7).shuffle(sh)
print(f"  z_g SHUF  AUC={np.mean([verify(Z,sh,st,seed=s) for s in (0,1,2)]):.4f}  (chance ~0.50)")
if has.sum()>500:
    Za=Z[has]; A=np.stack([r[3] for r in R if r[3] is not None]); pa=per[has]; sa=st[has]
    aaf=[verify(A,pa,sa,seed=s) for s in (0,1,2)]
    print(f"  AuraFace  AUC={np.mean(aaf):.4f}  seeds={[round(a,3) for a in aaf]}  (identity ceiling)")
    uids=sorted(set(pa.tolist())); rng=np.random.RandomState(0); rng.shuffle(uids)
    trp=set(uids[:int(0.8*len(uids))]); tr=np.array([p in trp for p in pa]); te=~tr
    sc=StandardScaler().fit(Za[tr]); m=Ridge(1.0).fit(sc.transform(Za[tr]),A[tr]); P=m.predict(sc.transform(Za[te]))
    r2=1-((A[te]-P)**2).sum()/((A[te]-A[tr].mean(0))**2).sum()
    print(f"  z_g->AuraFace person-split ridge R2={r2:+.3f}  (zero overlap ~ 0)")
print("\nDONE (Postgres approved set).")
