#!/usr/bin/env python3
"""
Phase 2: review.db-driven gate extractor (z_g + z_d).

Queries the hegre identity review database (READ-ONLY) for approved images on
contamination-free identities, encodes each through the frozen z_g production
encoder and per-mode z_d depth encoders, and caches the gate vectors to disk.

Usage:
  # Dry-run on 5 identities
  python scripts/20_extract_zd_gate.py --limit 5

  # Full gate extract
  python scripts/20_extract_zd_gate.py
"""
import os, sys, json, argparse
import numpy as np
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.zd_inference import encode_zd
from geometry_pca.constants import FACE_SLICE

MODES = ["A", "A_prime", "C"]
CONF_THRESH = 0.5
DATASET_SIGMA = 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N identities (0=all)")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESH)
    args = parser.parse_args()

    # ── 1. Load encoders ──────────────────────────────────────────────
    print("Loading encoders...")
    zg_enc = dict(np.load("output/encoder_production.npz"))
    zd_encs = {}
    for m in MODES:
        zd_encs[m] = dict(np.load(f"output/encoder_zd_{m}.npz"))
    print("  Done.")

    # ── 2. Query review.db (READ-ONLY) ───────────────────────────────
    db = sqlite3.connect("file:/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db?mode=ro", uri=True)
    c = db.cursor()
    c.execute("""
        SELECT i.image_path, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
        ORDER BY p.name, i.id
    """)
    all_rows = c.fetchall()
    db.close()

    # group by identity
    id_rows = {}
    for img_path, name, pid in all_rows:
        id_rows.setdefault(pid, []).append((img_path, name))

    identities = list(id_rows.items())
    limit_notice = f" (--limit {args.limit})" if args.limit else ""
    n_images_in_scope = sum(len(rows) for _, rows in identities)
    if args.limit > 0:
        identities = identities[:args.limit]
        n_images_in_scope = sum(len(rows) for _, rows in identities)

    print(f"\nGate set: {len(identities)} identities ({n_images_in_scope} images total){limit_notice}")

    # ── 3. Encode ─────────────────────────────────────────────────────
    zg_list, zd_lists = [], {m: [] for m in MODES}
    y_labels, y_names = [], []
    skip_by_id = {}
    n_ok, n_skip = 0, 0
    n_conf_skip = 0

    for pid, rows in identities:
        pid_ok, pid_skip = 0, 0
        for img_path, name in rows:
            # Map review.db path to face-tree path
            ed = os.path.join(
                "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum",
                ed_orig.split("hegre_enriched/", 1)[1]
            )
            try:
                pose = np.load(os.path.join(ed, "pose.npy")).astype(np.float32)
                depth = np.load(os.path.join(ed, "depth.npy")).astype(np.float32)
                seg = np.load(os.path.join(ed, "seg.npy"))
            except (FileNotFoundError, OSError, ValueError) as e:
                n_skip += 1; pid_skip += 1
                continue

            face = pose[FACE_SLICE]  # (68,3)

            # z_g: require mean confidence >= threshold
            if face[:, 2].mean() >= args.conf_thresh:
                z_g = encode_zg(face[:, :2], zg_enc)
            else:
                n_skip += 1; pid_skip += 1; n_conf_skip += 1
                continue

            # z_d: try all modes; if any mode fails, skip the image entirely
            # (keep per-mode vectors sample-aligned)
            zd_ok = True
            zd_vecs = {}
            for m in MODES:
                zd = encode_zd(depth, seg, face, zd_encs[m], mode=m,
                               dataset_sigma=DATASET_SIGMA)
                if zd is None:
                    zd_ok = False
                    break
                zd_vecs[m] = zd

            if not zd_ok:
                n_skip += 1; pid_skip += 1
                continue

            # all modes passed — accept
            zg_list.append(z_g)
            for m in MODES:
                zd_lists[m].append(zd_vecs[m])
            y_labels.append(pid)
            y_names.append(name)
            n_ok += 1; pid_ok += 1

        skip_by_id[pid] = {"ok": pid_ok, "skip": pid_skip}
        if pid_skip > 0:
            identity_label = id_rows[pid][0][1]
            print(f"  {identity_label:16s} kept {pid_ok:3d}  skipped {pid_skip:3d}")

    # ── 4. Stack and save ─────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    X_g = np.stack(zg_list).astype(np.float32) if zg_list else np.zeros((0, 50), np.float32)
    y_arr = np.array(y_labels, dtype=np.int32)
    names_arr = np.array(y_names)

    print(f"\nEncoded: {n_ok} images  skipped: {n_skip} "
          f"(conf<{args.conf_thresh}: {n_conf_skip})")
    print(f"  X_g shape: {X_g.shape}")

    for m in MODES:
        X_d = np.stack(zd_lists[m]).astype(np.float32) if zd_lists[m] else np.zeros((0, 50), np.float32)
        out = os.path.join("data", f"zd_gate_{m}.npz")
        np.savez_compressed(out, X_g=X_g, X_d=X_d, y=y_arr, names=names_arr,
                            skip_by_id=json.dumps(skip_by_id))
        print(f"  Saved {out}  X_d shape={X_d.shape}")

    with open(os.path.join("data", "zd_gate_meta.json"), "w") as f:
        json.dump({"n_identities": len(identities), "n_images_ok": n_ok,
                   "n_skip": n_skip, "n_conf_skip": n_conf_skip,
                   "conf_thresh": args.conf_thresh,
                   "skip_by_id": skip_by_id}, f, indent=2)
    print(f"  Meta saved to data/zd_gate_meta.json")


if __name__ == "__main__":
    main()
