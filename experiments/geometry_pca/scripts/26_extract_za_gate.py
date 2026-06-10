#!/usr/bin/env python3
"""
Phase 2b: review.db-driven gate extractor for z_a (normals).

Queries the hegre identity review database (READ-ONLY) for approved images on
contamination-free identities, encodes each through the frozen z_g production
encoder and per-variant z_a normal encoders, and caches the gate vectors.

Usage:
  --limit N    dry-run on first N identities
  --variant V  encode only this variant (otherwise all four)
"""
import os, sys, json, argparse
import numpy as np
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.za_inference import encode_za
from geometry_pca.constants import FACE_SLICE

VARIANTS = ["raw", "xy", "rot", "rot_xy"]
CONF_THRESH = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N identities")
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS,
                        help="Encode only this variant (default: all four)")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESH)
    args = parser.parse_args()

    variants = [args.variant] if args.variant else VARIANTS

    # Load encoders
    print("Loading encoders...")
    zg_enc = dict(np.load("output/encoder_production.npz"))
    za_encs = {}
    for v in variants:
        za_encs[v] = dict(np.load(f"output/encoder_za_{v}.npz"))
    print("  Done.")

    # Query review.db (READ-ONLY)
    db = sqlite3.connect("file:data/review.db?mode=ro", uri=True)
    c = db.cursor()
    c.execute("""
        SELECT i.enriched_dir, p.name, i.persona_id
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
          AND i.persona_id NOT IN (
              SELECT persona_id FROM images WHERE status = 'tainted:contamination'
          )
        ORDER BY p.name, i.id
    """)
    all_rows = c.fetchall()
    db.close()

    id_rows = {}
    for ed, name, pid in all_rows:
        id_rows.setdefault(pid, []).append((ed, name))

    identities = list(id_rows.items())
    if args.limit > 0:
        identities = identities[:args.limit]

    n_images_in_scope = sum(len(rows) for _, rows in identities)
    print(f"\nGate set: {len(identities)} identities ({n_images_in_scope} images total)")

    # Encode
    zg_list, za_lists = [], {v: [] for v in variants}
    y_labels, y_names = [], []
    skip_by_id = {}
    n_ok, n_skip, n_conf_skip = 0, 0, 0

    for pid, rows in identities:
        pid_ok, pid_skip = 0, 0
        for ed, name in rows:
            try:
                pose = np.load(os.path.join(ed, "pose.npy")).astype(np.float32)
                normal = np.load(os.path.join(ed, "normal.npy")).astype(np.float32)
                seg = np.load(os.path.join(ed, "seg.npy"))
            except (FileNotFoundError, OSError, ValueError):
                n_skip += 1; pid_skip += 1
                continue

            face = pose[FACE_SLICE]

            # z_g: mean confidence >= threshold
            if face[:, 2].mean() >= args.conf_thresh:
                z_g = encode_zg(face[:, :2], zg_enc)
            else:
                n_skip += 1; pid_skip += 1; n_conf_skip += 1
                continue

            # z_a per variant — if any variant fails, skip the image
            za_vecs = {}
            za_ok = True
            for v in variants:
                za = encode_za(normal, seg, face, za_encs[v], variant=v)
                if za is None:
                    za_ok = False
                    break
                za_vecs[v] = za

            if not za_ok:
                n_skip += 1; pid_skip += 1
                continue

            zg_list.append(z_g)
            for v in variants:
                za_lists[v].append(za_vecs[v])
            y_labels.append(pid)
            y_names.append(name)
            n_ok += 1; pid_ok += 1

        skip_by_id[pid] = {"ok": pid_ok, "skip": pid_skip}

    # Stack and save
    os.makedirs("data", exist_ok=True)
    X_g = np.stack(zg_list).astype(np.float32) if zg_list else np.zeros((0, 50), np.float32)
    y_arr = np.array(y_labels, dtype=np.int32)
    names_arr = np.array(y_names)

    print(f"\nEncoded: {n_ok} images  skipped: {n_skip} (conf<{args.conf_thresh}: {n_conf_skip})")
    print(f"  X_g shape: {X_g.shape}")

    for v in variants:
        X_a = np.stack(za_lists[v]).astype(np.float32) if za_lists[v] else np.zeros((0, 50), np.float32)
        out = os.path.join("data", f"za_gate_{v}.npz")
        np.savez_compressed(out, X_g=X_g, X_a=X_a, y=y_arr, names=names_arr,
                            skip_by_id=json.dumps(skip_by_id))
        print(f"  Saved {out}  X_a shape={X_a.shape}")

    with open(os.path.join("data", "za_gate_meta.json"), "w") as f:
        json.dump({"n_identities": len(identities), "n_images_ok": n_ok,
                   "n_skip": n_skip, "n_conf_skip": n_conf_skip,
                   "conf_thresh": args.conf_thresh,
                   "skip_by_id": skip_by_id}, f, indent=2)
    print(f"  Meta saved to data/za_gate_meta.json")


if __name__ == "__main__":
    main()
