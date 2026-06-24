#!/usr/bin/env python3
"""
Phase 2 face-crop seg-collapse control (2026-06-11 review).

CONTEXT: Sapiens body-part segmentation collapses on ~10% of tight face crops
(expects body context; extreme close-ups don't read as "person"). Stratum's
depth pass is seg-masked, so collapsed seg => near-empty depth.npy => identity-
free "empty" z_d vectors that show up as cross-identity near-duplicates
(e.g. gislane≈vika cos 0.994). Visually confirmed: perfect frontal face crops
with seg foreground <1%.

This control re-runs the z_d verification-AUC gate EXCLUDING all rows whose
face-crop seg foreground fraction is below FG_MIN. Result (2026-06-11): the
FAIL *strengthened* (best delta −0.023 -> −0.034), proving the seg defect was
not masking a PASS.

⚠️ normal.npy is masked by the same seg — Phase 2b (z_a) on face crops MUST
apply this same filter.

Reads: data/zd_gate_{A,A_prime,C}.npz (from 20_extract_zd_gate.py),
       review.db (READ-ONLY), face-tree seg.npy/pose.npy.
Writes: data/zd_facecrop_seg_control.json
"""
import os, sys, json, sqlite3
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc, partition_gate
from geometry_pca.zg_inference import encode_zg
from geometry_pca.constants import FACE_SLICE

FACE_TREE = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
MODES = ["A", "A_prime", "C"]
SEEDS = [0, 1, 2]
FG_MIN = 0.30
CONF_THRESH = 0.5


def main():
    zg_enc = dict(np.load("output/encoder_production.npz"))

    db = sqlite3.connect("file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro", uri=True)
    rows = db.execute("""
        SELECT i.enriched_dir, p.name FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination')
        ORDER BY p.name, i.id
    """).fetchall()
    db.close()

    z0 = np.load(f"data/zd_gate_{MODES[0]}.npz", allow_pickle=True)
    Xg_art, names_art = z0["X_g"], z0["names"]

    # Match artifact rows to leaves via z_g equality (names bucket to cut comparisons)
    art_by_name = defaultdict(list)
    for i, nm in enumerate(names_art):
        art_by_name[str(nm)].append(i)

    fg_per_row = np.full(len(Xg_art), np.nan)
    n_matched = 0
    for img_path, name in rows:
        cand = art_by_name.get(name)
        if not cand:
            continue
        leaf = ed_orig.split("hegre_enriched/", 1)[1]
        ed = os.path.join(FACE_TREE, leaf)
        try:
            pose = np.load(os.path.join(ed, "pose.npy")).astype(np.float32)
            seg = np.load(os.path.join(ed, "seg.npy"))
        except (FileNotFoundError, OSError, ValueError):
            continue
        face = pose[FACE_SLICE]
        if face[:, 2].mean() < CONF_THRESH:
            continue
        zg = encode_zg(face[:, :2], zg_enc)
        fgv = float((seg > 0).mean())
        for i in cand:
            if np.isnan(fg_per_row[i]) and np.allclose(zg, Xg_art[i], atol=1e-5):
                fg_per_row[i] = fgv
                n_matched += 1
                break

    ok = ~np.isnan(fg_per_row)
    keep = ok & (fg_per_row >= FG_MIN)
    print(f"matched {n_matched}/{len(Xg_art)} rows; fg>={FG_MIN:.0%}: {keep.sum()} "
          f"(dropped {int(ok.sum()) - int(keep.sum())} low-fg, {int((~ok).sum())} unmatched)")

    results = {"fg_min": FG_MIN, "n_total": int(len(Xg_art)),
               "n_kept": int(keep.sum()), "modes": {}}
    print(f"\nGATE RE-RUN ON SEG-CLEAN SUBSET (fg >= {FG_MIN:.0%})")
    print("=" * 60)
    first = True
    for mode in MODES:
        zz = np.load(f"data/zd_gate_{mode}.npz", allow_pickle=True)
        Xg, Xd, y = zz["X_g"][keep], zz["X_d"][keep], zz["y"][keep]
        if first:
            auc_g = float(np.mean([verification_auc(Xg, y, seed=s)[0] for s in SEEDS]))
            results["n_identities"] = int(len(np.unique(y)))
            results["zg_baseline_auc"] = auc_g
            print(f"  set: {keep.sum()} images, {len(np.unique(y))} identities")
            print(f"  z_g baseline AUC = {auc_g:.4f}")
            first = False
        auc_d = float(np.mean([verification_auc(Xd, y, seed=s)[0] for s in SEEDS]))
        cat = float(np.mean([verification_auc(np.hstack([Xg, Xd]), y, seed=s)[0] for s in SEEDS]))
        delta = float(np.mean([partition_gate(Xg, Xd, y, eps=0.01, seed=s)["delta"] for s in SEEDS]))
        verdict = "PASS" if delta > 0.01 else "FAIL"
        print(f"  Mode {mode}: z_d alone={auc_d:.4f}  [z_g|z_d]={cat:.4f}  "
              f"delta={delta:+.4f} -> {verdict}")
        results["modes"][mode] = {"zd_alone_auc": auc_d, "concat_auc": cat,
                                  "delta": delta, "verdict": verdict}

    overall = "PASS" if any(m["verdict"] == "PASS" for m in results["modes"].values()) else "FAIL"
    results["overall_verdict"] = overall
    print(f"\nOVERALL (seg-clean): {overall}")

    with open("data/zd_facecrop_seg_control.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved data/zd_facecrop_seg_control.json")


if __name__ == "__main__":
    main()
