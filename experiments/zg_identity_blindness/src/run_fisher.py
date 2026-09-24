#!/usr/bin/env python3
"""zg-identity-blindness — rebuilt Fisher-J instrument for z_g's identity content.

Pre-registered in ../README.md (written 2026-09-24, before the first run).
Mode: confirmatory. Declared in ../provenance.yaml before run 1.

The metric is the GLOBAL Fisher discriminant ratio, matching the lost original:

    J = S_B / S_W
    S_B = sum_c n_c * ||mu_c - mu||^2 / N      (between-identity scatter)
    S_W = sum_c sum_i ||z_i - mu_c||^2 / N     (within-identity scatter)

implemented by geometry_pca.fisher.fisher_ratios (the same function the legacy
gate sweep used). Both scatters are always printed: a high J achieved by a
collapsed S_B is not identity separability.

Subcommands: g0 | g1 | g2 | g3 | all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# paths (verified 2026-09-24)
# ---------------------------------------------------------------------------
CORPUS = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"
SOURCE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"
ZG_SRC = "/mnt/nas-ai-models/training-data/eidolon/hegre_corpus"  # corpus z_g is the consumed copy

# The zg source tree used by the zg-validity arm (flat {image_id}.npy):
ZG_TREE = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/zg"

FACE_SLICE = slice(23, 91)  # 68 face keypoints of the 133 DWPose whole-body

# repo import of the shared Fisher implementation (the original metric)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "experiments", "geometry_pca"))
from geometry_pca.fisher import fisher_ratios, restandardize  # noqa: E402

VENUES = {"A": 0.0, "B": 0.3, "C": 0.5}  # mean face-keypoint confidence floors
PRIMARY_VENUE = "B"


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------
def iter_samples():
    """Yield (sample_dir_name, persona, set, image_id) for every corpus sample."""
    for name in sorted(os.listdir(CORPUS)):
        d = os.path.join(CORPUS, name)
        if not os.path.isdir(d):
            continue
        meta_p = os.path.join(d, "metadata.json")
        if not os.path.exists(meta_p):
            continue
        with open(meta_p) as fh:
            m = json.load(fh)
        yield name, m["persona"], m["set"], m["image_id"]


def face_conf(image_id: str, persona: str, set_: str) -> float:
    """Mean confidence over the 68 face keypoints, read from the SOURCE pose.npy.

    Returns -1.0 when the pose file is absent (counted separately, never silently 0).
    """
    p = os.path.join(SOURCE, "stratum", persona, set_, f"{image_id}.npy")
    if not os.path.exists(p):
        p = os.path.join(SOURCE, "stratum", persona, set_, image_id, "pose.npy")
    if not os.path.exists(p):
        return -1.0
    po = np.load(p)
    return float(np.asarray(po[FACE_SLICE, 2], dtype=np.float64).mean())


def load_corpus(limit=None):
    """Load z_g, auraface_lda, persona and face-confidence for every sample."""
    zgs, afs, personas, confs, names = [], [], [], [], []
    n_missing_zg = n_missing_af = n_missing_pose = 0
    for i, (name, persona, set_, image_id) in enumerate(iter_samples()):
        if limit is not None and i >= limit:
            break
        zg_p = os.path.join(CORPUS, name, "z_g.npy")
        af_p = os.path.join(CORPUS, name, "auraface_lda.npy")
        if not os.path.exists(zg_p):
            n_missing_zg += 1
            continue
        if not os.path.exists(af_p):
            n_missing_af += 1
        zgs.append(np.load(zg_p).astype(np.float64))
        afs.append(np.load(af_p).astype(np.float64) if os.path.exists(af_p) else np.full(64, np.nan))
        personas.append(persona)
        c = face_conf(image_id, persona, set_)
        if c < 0:
            n_missing_pose += 1
        confs.append(c)
        names.append(name)

    log(f"  loaded {len(zgs)} samples | missing z_g {n_missing_zg} | "
        f"missing auraface {n_missing_af} | missing pose {n_missing_pose}")
    return (np.stack(zgs), np.stack(afs), np.array(personas),
            np.array(confs, dtype=np.float64), names)


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def j_of(Z, y):
    """Global Fisher J + the scatters. Returns (J, S_B, S_W, J_Ci)."""
    J, S_B, S_W, J_Ci, _, _ = fisher_ratios(Z, y)
    return float(J), float(S_B), float(S_W), np.asarray(J_Ci)


def keep_min2(Z, y):
    """Drop identities with < 2 samples (within-scatter undefined) and count them."""
    counts = defaultdict(int)
    for lab in y:
        counts[lab] += 1
    keep = np.array([counts[lab] >= 2 for lab in y])
    dropped_ids = int(sum(1 for v in counts.values() if v < 2))
    return Z[keep], y[keep], dropped_ids


def summarise(tag, Z, y, Z_rs=None):
    """Print and return the J block for one (array, venue)."""
    Zk, yk, dropped = keep_min2(Z, y)
    J, S_B, S_W, J_Ci = j_of(Zk, yk)
    n_ident = len(set(yk.tolist()))
    n_morph = int((J_Ci > 0.15).sum())
    n_trans = int((J_Ci < 0.05).sum())
    row = dict(tag=tag, J=J, S_B=S_B, S_W=S_W, n_samples=int(len(Zk)),
               n_identities=n_ident, dropped_ids=dropped,
               n_morph_axes=n_morph, n_transient_axes=n_trans)
    if Z_rs is not None:
        J2, S_B2, S_W2, J_Ci2 = j_of(*keep_min2(Z_rs, y)[:2])
        row.update(J_restandardized=float(J2), S_B_rs=float(S_B2), S_W_rs=float(S_W2),
                   n_morph_axes_rs=int((J_Ci2 > 0.15).sum()))
    log(f"  {tag:34s} J={J:8.4f}  S_B={S_B:10.4f}  S_W={S_W:10.4f}  "
        f"morph(J>0.15)={n_morph:3d}  trans(J<0.05)={n_trans:3d}  "
        f"n={len(Zk)} ids={n_ident} dropped_ids={dropped}")
    if Z_rs is not None:
        log(f"  {tag + ' [restandardized]':34s} J={row['J_restandardized']:8.4f}  "
            f"morph(J>0.15)={row['n_morph_axes_rs']:3d}")
    return row


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def cmd_g0(args):
    """G0 — instrument identity: corpus z_g bit-identical to the encoder source."""
    log("=== G0 — instrument identity (corpus z_g vs source) ===")
    rng = np.random.RandomState(20260924)
    samples = list(iter_samples())
    pick = [samples[i] for i in rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)]
    worst = 0.0
    missing = 0
    checked = 0
    for name, persona, set_, image_id in pick:
        cand = [
            os.path.join(ZG_TREE, "faces", persona, set_, f"{image_id}.npy"),
            os.path.join(ZG_TREE, f"{image_id}.npy"),
        ]
        src = next((c for c in cand if os.path.exists(c)), None)
        if src is None:
            missing += 1
            continue
        a = np.load(os.path.join(CORPUS, name, "z_g.npy")).astype(np.float64)
        b = np.load(src).astype(np.float64)
        worst = max(worst, float(np.abs(a - b).max()))
        checked += 1
    log(f"  checked {checked} | missing source {missing} | max|diff| = {worst:.10e}")
    passed = (missing == 0) and (worst == 0.0) and checked > 0
    if missing:
        log("  NOTE: source tree layout may differ; if 'missing' equals the full sample,")
        log("        the comparison is void, not a pass.")
    log(f"  G0 {'PASS' if passed else 'FAIL'}")
    return dict(gate="G0", checked=checked, missing_source=missing,
                max_abs_diff=worst, passed=bool(passed))


def cmd_g1(args, cache=None):
    """G1 — positive control: the instrument MUST separate AuraFace identity."""
    log("=== G1 — positive control (AuraFace must show higher J than z_g) ===")
    zg, af, personas, confs, _ = cache or load_corpus(args.limit)
    keep = np.isfinite(af).all(1)
    zg, af, personas, confs = zg[keep], af[keep], personas[keep], confs[keep]
    log(f"  usable {len(zg)} samples")
    # fair comparison: restandardize both so per-component scale is matched
    zg_rs, af_rs = restandardize(zg), restandardize(af)
    j_zg = summarise("z_g", zg, personas, zg_rs)
    j_af = summarise("auraface_lda", af, personas, af_rs)
    ratio_raw = j_af["J"] / max(j_zg["J"], 1e-12)
    ratio_rs = j_af["J_restandardized"] / max(j_zg["J_restandardized"], 1e-12)
    log(f"  ratio J_auraface/J_zg  raw={ratio_raw:.2f}x  restandardized={ratio_rs:.2f}x")
    passed = ratio_rs >= 3.0 or ratio_raw >= 3.0
    log(f"  G1 {'PASS' if passed else 'FAIL — instrument cannot detect identity separability; arm is VOID'}")
    return dict(gate="G1", z_g=j_zg, auraface=j_af,
                ratio_raw=float(ratio_raw), ratio_rs=float(ratio_rs), passed=bool(passed))


def cmd_g2(args, cache=None):
    """G2 — the headline re-measurement across the venue family."""
    log("=== G2 — headline re-measurement on the curated corpus ===")
    zg, af, personas, confs, _ = cache or load_corpus(args.limit)
    n_total = len(zg)
    log(f"  corpus samples consumed: {n_total} | personas: {len(set(personas.tolist()))}")
    log(f"  face-confidence: min={np.nanmin(confs):.3f} "
        f"p05={np.nanpercentile(confs, 5):.3f} median={np.nanmedian(confs):.3f} "
        f"| no pose: {int((confs < 0).sum())}")
    out = {}
    for vname, floor in VENUES.items():
        sel = confs >= floor if floor > 0 else np.ones_like(confs, dtype=bool)
        if vname != "A":
            sel &= confs >= 0
        log(f"\n  --- venue {vname} (mean face conf >= {floor}) : {int(sel.sum())} samples ---")
        out[vname] = summarise(f"z_g venue {vname}", zg[sel], personas[sel],
                               restandardize(zg[sel]))
    prim = out[PRIMARY_VENUE]
    J = prim["J"]
    morph = prim["n_morph_axes"]
    if 0.02 <= J <= 0.12 and morph <= 10:
        verdict = "CONFIRM"
    elif J >= 0.20 or morph >= 20:
        verdict = "FALSIFY"
    else:
        verdict = "PARTIAL"
    log(f"\n  PRIMARY venue {PRIMARY_VENUE}: J={J:.4f}  morph={morph}")
    log(f"  reference: original (lost script, pre-curation corpus) J=0.059, morph 6 (was 27)")
    log(f"  G2 verdict: {verdict}")
    return dict(gate="G2", venues=out, primary_venue=PRIMARY_VENUE, J=J,
                n_morph_axes=morph, verdict=verdict,
                reference_J=0.059, reference_note="lost script; pre-curation 69,110/323 corpus")


def cmd_g3(args, cache=None):
    """G3 — legacy collapse reproducibility (honest UNREPRODUCIBLE if unreconstructible)."""
    log("=== G3 — legacy 27->6 collapse reproducibility ===")
    log("  The original selection (which 1,448 / 101 subset, which tier) is not")
    log("  reconstructible: the producing script is absent from every branch.")
    log("  Verdict: UNREPRODUCIBLE — no stand-in is substituted.")
    return dict(gate="G3", verdict="UNREPRODUCIBLE",
                reason="original producing script absent from all branches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["g0", "g1", "g2", "g3", "all"])
    ap.add_argument("--limit", type=int, default=None, help="cap samples (debug only)")
    ap.add_argument("--n", type=int, default=400, help="G0 sample size")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    results = {}
    cache = None
    if args.cmd in ("g1", "g2", "g3", "all"):
        log("loading corpus...")
        cache = load_corpus(args.limit)
    if args.cmd in ("g0", "all"):
        results["g0"] = cmd_g0(args)
    if args.cmd in ("g1", "all"):
        results["g1"] = cmd_g1(args, cache)
    if args.cmd in ("g2", "all"):
        results["g2"] = cmd_g2(args, cache)
    if args.cmd in ("g3", "all"):
        results["g3"] = cmd_g3(args, cache)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
