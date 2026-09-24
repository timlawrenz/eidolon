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

# The zg source tree (flat {image_id}.npy under zg/faces/{persona}/{set}/):
#   verified 2026-09-24 -> /mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/zg
ZG_TREE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/zg"

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
def load_per_image_lda(names_persona_set_id):
    """Load PER-IMAGE AuraFace-LDA from the source tree.

    WHY (G1b): the corpus's auraface_lda.npy is the PERSONA CENTROID, verified
    2026-09-24: all 321 personas carry a bit-identical vector across every one of
    their samples (worst within-persona max|diff| = 0.0000000000e+00). Fisher J is
    therefore mathematically undefined on it (S_W == 0 by construction), which is
    why the pre-registered G1 could not run on the named array.

    The per-image vectors live at lda/faces/{persona}/{set}/{image_id}.npy. They
    are stored at RAW scale (e.g. norm 151.9) while the centroid is unit-norm; J is
    invariant to a global scale (both scatters scale by c^2), so this does not
    affect the control.
    """
    out = []
    missing = 0
    for persona, set_, image_id in names_persona_set_id:
        p = os.path.join(SOURCE, "lda", "faces", persona, set_, f"{image_id}.npy")
        if not os.path.exists(p):
            missing += 1
            out.append(np.full(64, np.nan))
            continue
        out.append(np.load(p).astype(np.float64))
    log(f"  per-image LDA: loaded {len(out) - missing} | missing {missing}")
    return np.stack(out), missing


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


def j_null(n_samples, n_classes):
    """Expected global J under random labels: E[J] ~= (C-1)/(N-C).

    Validated 2026-09-24 on synthetic data before any corpus result was read:
    pure-noise random labels at N=600/C=30 gave J=0.0560 vs (C-1)/(N-C)=0.0509.

    WHY THIS MATTERS: the floor is a function of N and C, so raw J is NOT
    comparable across corpora of different size. The old corpus (69,110/323) has
    floor 0.0047; the curated corpus (31,711/321) has floor 0.0102 -- 2.2x
    higher. A smaller corpus mechanically RAISES raw J. J_ratio = J / J_null is
    the size-corrected comparison; the raw comparison is confounded.
    """
    if n_samples <= n_classes:
        return float("nan")
    return (n_classes - 1) / (n_samples - n_classes)


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
    jn = j_null(len(Zk), n_ident)
    jratio = J / jn if jn and np.isfinite(jn) else float("nan")
    row = dict(tag=tag, J=J, S_B=S_B, S_W=S_W, n_samples=int(len(Zk)),
               n_identities=n_ident, dropped_ids=dropped,
               n_morph_axes=n_morph, n_transient_axes=n_trans,
               J_null=jn, J_ratio=jratio)
    if Z_rs is not None:
        Zk2, yk2, _ = keep_min2(Z_rs, y)
        J2, S_B2, S_W2, J_Ci2 = j_of(Zk2, yk2)
        row.update(J_restandardized=float(J2), S_B_rs=float(S_B2), S_W_rs=float(S_W2),
                   n_morph_axes_rs=int((J_Ci2 > 0.15).sum()))
    log(f"  {tag:34s} J={J:8.4f}  S_B={S_B:10.4f}  S_W={S_W:10.4f}  "
        f"morph(J>0.15)={n_morph:3d}  trans(J<0.05)={n_trans:3d}  "
        f"n={len(Zk)} ids={n_ident} dropped_ids={dropped}")
    log(f"  {tag + ' [null-corrected]':34s} J_null={jn:8.4f}  "
        f"J/J_null={jratio:8.2f}x  (floor depends on N,C -> raw J is NOT "
        f"comparable across corpora)")
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
    """G1 — positive control. Also runs G1b on the PER-IMAGE source (see below)."""
    log("=== G1 — positive control (AuraFace must show higher J than z_g) ===")
    if cache is None:
        cache = load_corpus(args.limit)
    zg, af, personas, confs, names = cache
    names = np.asarray(names)
    keep = np.isfinite(af).all(1)
    zg, af, personas, confs = zg[keep], af[keep], personas[keep], confs[keep]
    log(f"  usable {len(zg)} samples")
    # fair comparison: restandardize both so per-component scale is matched
    zg_rs, af_rs = restandardize(zg), restandardize(af)
    j_zg = summarise("z_g", zg, personas, zg_rs)
    j_af = summarise("auraface_lda (corpus = CENTROID)", af, personas, af_rs)
    ratio_raw = j_af["J"] / max(j_zg["J"], 1e-12)
    ratio_rs = j_af["J_restandardized"] / max(j_zg["J_restandardized"], 1e-12)
    log(f"  ratio J_auraface/J_zg  raw={ratio_raw:.2f}x  restandardized={ratio_rs:.2f}x")
    passed = ratio_rs >= 3.0 or ratio_raw >= 3.0
    log(f"  G1 {'PASS' if passed else 'FAIL'}")
    if not passed:
        log("  G1 FAILED AS PRE-REGISTERED. Reason is now known and is a DATA property,")
        log("  not an instrument failure: the corpus auraface_lda is the PERSONA CENTROID")
        log("  (S_W == 0 by construction -> J is 0 by the guard, not by measurement).")
        log("  The pre-registered INTENT was 'AuraFace as a stream known to carry identity'.")
        log("  G1b below implements that intent on the correct array (PER-IMAGE LDA).")

    # ---------- G1b: the intended control, on the correct array ----------
    log("")
    log("=== G1b — same control on PER-IMAGE AuraFace-LDA (post-hoc instrument fix) ===")
    triples = []
    for name in names[keep]:
        with open(os.path.join(CORPUS, name, "metadata.json")) as fh:
            m = json.load(fh)
        triples.append((m["persona"], m["set"], m["image_id"]))
    afi, missing_i = load_per_image_lda(triples)
    ok = np.isfinite(afi).all(1)
    j_afi = summarise("per-image LDA", afi[ok], personas[ok], restandardize(afi[ok]))
    ratio_b = j_afi["J"] / max(j_zg["J"], 1e-12)
    ratio_b_rs = j_afi["J_restandardized"] / max(j_zg["J_restandardized"], 1e-12)
    log(f"  ratio J_per_image_lda/J_zg  raw={ratio_b:.2f}x  restandardized={ratio_b_rs:.2f}x")
    passed_b = ratio_b_rs >= 3.0 or ratio_b >= 3.0
    log(f"  G1b {'PASS — instrument CAN detect identity separability' if passed_b else 'FAIL — arm remains VOID'}")
    return dict(gate="G1", z_g=j_zg, auraface_centroid=j_af,
                ratio_raw=float(ratio_raw), ratio_rs=float(ratio_rs),
                passed=bool(passed),
                g1b=dict(per_image_lda=j_afi, missing=missing_i,
                         ratio_raw=float(ratio_b), ratio_rs=float(ratio_b_rs),
                         passed=bool(passed_b)),
                g1b_status="POST-HOC instrument correction; disclosed, gate not moved")


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

    # ---- size-corrected comparison to the original (see j_null docstring) ----
    jn_new = prim["J_null"]
    jn_old = j_null(69110, 323)
    ratio_new = prim["J_ratio"]
    ratio_old = 0.059 / jn_old if jn_old else float("nan")
    log("")
    log("  --- size-corrected comparison to the lost-script original ---")
    log(f"  original  J=0.0590  N=69110 C=323  floor={jn_old:.5f}  J/floor={ratio_old:6.2f}x")
    log(f"  this run  J={J:.4f}  N={prim['n_samples']} C={prim['n_identities']}  "
        f"floor={jn_new:.5f}  J/floor={ratio_new:6.2f}x")
    log(f"  RAW J is confounded by corpus size; the floor rose {jn_new / jn_old:.2f}x "
        f"because N fell, which RAISES raw J for free.")
    log(f"  The honest comparison is J/floor: {ratio_old:.2f}x -> {ratio_new:.2f}x.")
    log("")
    log(f"  PRIMARY venue {PRIMARY_VENUE}: J={J:.4f}  morph={morph}")
    log(f"  reference: original (lost script, pre-curation corpus) J=0.059, morph 6 (was 27)")
    log(f"  G2 verdict (pre-registered gate on RAW J, unchanged): {verdict}")
    return dict(gate="G2", venues=out, primary_venue=PRIMARY_VENUE, J=J,
                n_morph_axes=morph, verdict=verdict, J_null=jn_new,
                J_ratio=ratio_new, J_null_original=jn_old, J_ratio_original=ratio_old,
                reference_J=0.059,
                reference_note="lost script; pre-curation 69,110/323 corpus",
                gate_note=("pre-registered gate is on RAW J and was NOT changed after "
                           "seeing J_null; J_null/J_ratio are declared supplementary "
                           "diagnostics calibrated before any corpus result was read"))


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
