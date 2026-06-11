#!/usr/bin/env python3
"""
Phase 2b SYSTEMATIC REVIEW: code, data, and results verification for the z_a PASS.

Sections:
  A. CODE — independent AUC cross-check (sklearn) + permutation null
  B. DATA — alignment guards, cross-phase X_g equality, Sapiens Y-convention
  C. RESULTS — 10-seed stats, rot-paradox mechanism test, per-identity
     robustness, shoot-leakage risk quantification (review.db, READ-ONLY)
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc, partition_gate
from geometry_pca.fisher import restandardize

VARIANTS = ["raw", "xy", "rot", "rot_xy"]
RESULTS = {}


def yaw_pitch_from_R(R):
    yaw = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
    return yaw, pitch


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ───────────────────────── A. CODE ─────────────────────────
section("A1. AUC IMPLEMENTATION CROSS-CHECK (ours vs sklearn)")
from sklearn.metrics import roc_auc_score

d = np.load("data/za_gate_raw.npz")
Xg, Xa, y = d["X_g"], d["X_a"], d["y"]

# rebuild the EXACT pairs our verification_auc uses (same RNG path), score with sklearn
def pairs_and_scores(Z, y, n_pairs=40000, seed=0):
    rng = np.random.default_rng(seed)
    Zs = restandardize(Z)
    Zn = Zs / np.maximum(np.linalg.norm(Zs, axis=1, keepdims=True), 1e-8)
    idx_by_id = {}
    for i, lab in enumerate(y):
        idx_by_id.setdefault(int(lab), []).append(i)
    multi = [k for k, v in idx_by_id.items() if len(v) >= 2]
    all_ids = list(idx_by_id.keys())
    half = n_pairs // 2
    sims, labels = [], []
    for _ in range(half):
        cid = multi[rng.integers(len(multi))]
        a, b = rng.choice(idx_by_id[cid], size=2, replace=False)
        sims.append(float(Zn[a] @ Zn[b])); labels.append(1)
    for _ in range(half):
        c1, c2 = rng.choice(all_ids, size=2, replace=False)
        a = idx_by_id[c1][rng.integers(len(idx_by_id[c1]))]
        b = idx_by_id[c2][rng.integers(len(idx_by_id[c2]))]
        sims.append(float(Zn[a] @ Zn[b])); labels.append(0)
    return np.array(labels), np.array(sims)

ours, _, _ = verification_auc(Xg, y, seed=0)
lab, sim = pairs_and_scores(Xg, y, seed=0)
sk = roc_auc_score(lab, sim)
print(f"  z_g  ours={ours:.6f}  sklearn(same pairs)={sk:.6f}  diff={abs(ours-sk):.2e}")
RESULTS["auc_crosscheck_diff"] = abs(ours - sk)

section("A2. PERMUTATION NULL (shuffled identities -> delta must be ~0)")
rng = np.random.default_rng(123)
null_deltas = []
for p in range(3):
    y_perm = rng.permutation(y)
    g = partition_gate(Xg, Xa, y_perm, seed=p)
    null_deltas.append(g["delta"])
    print(f"  perm {p}: baseline={g['auc_baseline']:.4f} cat={g['auc_concatenated']:.4f} delta={g['delta']:+.4f}")
RESULTS["permutation_null_max_abs_delta"] = float(np.max(np.abs(null_deltas)))

# ───────────────────────── B. DATA ─────────────────────────
section("B1. ALIGNMENT: all 4 za files share names/y/X_g; run stamps equal")
ref = np.load("data/za_gate_raw.npz")
ok = True
for v in VARIANTS[1:]:
    dv = np.load(f"data/za_gate_{v}.npz")
    same_names = np.array_equal(dv["names"], ref["names"])
    same_y = np.array_equal(dv["y"], ref["y"])
    same_xg = np.array_equal(dv["X_g"], ref["X_g"])
    same_stamp = str(dv["run_stamp"]) == str(ref["run_stamp"])
    ok &= same_names and same_y and same_xg and same_stamp
    print(f"  {v:8s} names={same_names} y={same_y} X_g={same_xg} stamp={same_stamp}")
RESULTS["za_files_aligned"] = bool(ok)

section("B2. CROSS-PHASE: za X_g vs zd X_g (same image set -> exact comparability?)")
try:
    zd = np.load("data/zd_gate_A.npz")
    same_n = len(zd["X_g"]) == len(ref["X_g"])
    same_names = np.array_equal(zd["names"], ref["names"]) if same_n else False
    same_xg = np.array_equal(zd["X_g"], ref["X_g"]) if same_n else False
    print(f"  n_equal={same_n}  names_equal={same_names}  X_g_equal={same_xg}")
    RESULTS["cross_phase_same_gate_set"] = bool(same_names and same_xg)
except FileNotFoundError:
    print("  zd gate files not found — skipped")

section("B3. SAPIENS NORMAL Y-CONVENTION (empirical, 2000 FFHQ cache samples)")
raw_cache = np.load("data/normal_cache/ffhq_normal_raw.npy", mmap_mode="r")
sub = np.asarray(raw_cache[:2000])          # (2000,64,64,3)
mag = np.linalg.norm(sub, axis=-1)
valid = mag > 0.1
ny_sum = np.where(valid, sub[..., 1], 0.0).sum(axis=0)   # (64,64) summed over samples
cnt = valid.sum(axis=0)
ny_mean = np.divide(ny_sum, cnt, out=np.zeros_like(ny_sum), where=cnt > 0)
top_rows = ny_mean[4:16, :].mean()     # forehead region (rows are +Y-down image order)
bot_rows = ny_mean[48:60, :].mean()    # chin/neck region
print(f"  mean ny forehead rows[4:16] = {top_rows:+.4f}")
print(f"  mean ny chin rows[48:60]    = {bot_rows:+.4f}")
conv = "+Y UP (OpenGL-style)" if top_rows > bot_rows else "+Y DOWN (image-style)"
print(f"  forehead tilts 'up' -> if ny(forehead) > ny(chin): normals are {conv}")
RESULTS["ny_forehead"] = float(top_rows); RESULTS["ny_chin"] = float(bot_rows)
RESULTS["inferred_normal_convention"] = conv

# ───────────────────────── C. RESULTS ─────────────────────────
section("C1. 10-SEED DELTA STATS (tighter than the 3-seed gate run)")
seed_list = list(range(10))
deltas_by_v = {}
for v in VARIANTS:
    Xa_v = np.load(f"data/za_gate_{v}.npz")["X_a"]
    ds = [partition_gate(Xg, Xa_v, y, seed=s)["delta"] for s in seed_list]
    deltas_by_v[v] = ds
    print(f"  {v:8s} mean={np.mean(ds):+.4f}  std={np.std(ds):.4f}  min={np.min(ds):+.4f}  (all>0.01: {all(d>0.01 for d in ds)})")
RESULTS["deltas_10seed"] = {v: list(map(float, ds)) for v, ds in deltas_by_v.items()}

print("\n  paired rot-vs-xy per-seed difference:")
diff = np.array(deltas_by_v["rot"]) - np.array(deltas_by_v["xy"])
print(f"  mean={diff.mean():+.4f}  std={diff.std():.4f}  rot>xy in {int((diff>0).sum())}/10 seeds")
RESULTS["rot_minus_xy"] = {"mean": float(diff.mean()), "std": float(diff.std())}

section("C2. ROT-PARADOX MECHANISM TEST (visibility bias -> pose injection)")
# Hypothesis: raw normals' MEAN direction is camera-facing regardless of pose
# (visible surfaces face the camera). After R^T de-rotation, the mean direction
# becomes R^T @ z_cam — i.e. it DIRECTLY encodes head pose. PCA then promotes
# this global pose direction into top components -> higher pose corr for rot.
rots = np.load("data/normal_cache/rotations.npy", mmap_mode="r")
N = 5000
sub = np.asarray(raw_cache[:N]); R_sub = np.asarray(rots[:N])
mag = np.linalg.norm(sub, axis=-1); valid = (mag > 0.1)[..., None]
mean_raw = (sub * valid).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)  # (N,3)
yaws = np.array([yaw_pitch_from_R(R_sub[i])[0] for i in range(N)])
# raw: mean normal x-component vs yaw
r_raw = np.corrcoef(mean_raw[:, 0], yaws)[0, 1]
# rot: de-rotated mean normal x-component vs yaw
mean_rot = np.einsum("nij,nj->ni", np.transpose(R_sub, (0, 2, 1)), mean_raw)
r_rot = np.corrcoef(mean_rot[:, 0], yaws)[0, 1]
print(f"  corr( mean_raw_nx , yaw ) = {r_raw:+.3f}   (raw: camera-facing bias ~ pose-blind mean)")
print(f"  corr( mean_rot_nx , yaw ) = {r_rot:+.3f}   (rot: de-rotated mean encodes pose)")
print(f"  -> hypothesis {'CONFIRMED' if abs(r_rot) > abs(r_raw) + 0.3 else 'NOT confirmed'}")
RESULTS["paradox_corr_raw"] = float(r_raw); RESULTS["paradox_corr_rot"] = float(r_rot)

section("C3. PER-IDENTITY ROBUSTNESS (drop 10 largest identities, recompute)")
ids_, counts = np.unique(y, return_counts=True)
big10 = ids_[np.argsort(counts)[-10:]]
keep = ~np.isin(y, big10)
print(f"  dropping {len(big10)} largest identities -> {keep.sum()} images, {len(np.unique(y[keep]))} identities remain")
for v in ["xy", "rot"]:
    Xa_v = np.load(f"data/za_gate_{v}.npz")["X_a"]
    g = partition_gate(Xg[keep], Xa_v[keep], y[keep], seed=0)
    print(f"  {v:8s} delta(without big10) = {g['delta']:+.4f}  (full-set seed0: {deltas_by_v[v][0]:+.4f})")

section("C4. SHOOT-LEAKAGE RISK (review.db READ-ONLY: sets per identity)")
import sqlite3
db = sqlite3.connect("file:data/review.db?mode=ro", uri=True)
rows = db.execute("""
    SELECT p.name, COUNT(DISTINCT i.set_id) AS n_sets, COUNT(*) AS n_imgs
    FROM images i JOIN personas p ON i.persona_id = p.id
    WHERE i.status='approved'
      AND i.persona_id NOT IN (SELECT persona_id FROM images WHERE status='tainted:contamination')
    GROUP BY i.persona_id
""").fetchall()
db.close()
n_sets = np.array([r[1] for r in rows])
print(f"  identities: {len(rows)}  | sets/identity: min={n_sets.min()} median={int(np.median(n_sets))} max={n_sets.max()}")
single_set = int((n_sets == 1).sum())
print(f"  identities with only ONE set: {single_set}/{len(rows)}")
print(f"  -> same-identity pairs {'CAN cross sets for most identities' if np.median(n_sets) >= 2 else 'mostly WITHIN one set — leakage risk HIGH'}")
RESULTS["sets_per_identity_median"] = float(np.median(n_sets))
RESULTS["single_set_identities"] = single_set

with open("data/za_systematic_review.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=float)
print(f"\nSaved data/za_systematic_review.json")
