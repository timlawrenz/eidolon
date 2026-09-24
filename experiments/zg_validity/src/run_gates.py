#!/usr/bin/env python3
"""zg-validity-threshold — gate runner.

Arm: experiments/zg_validity/  |  branch: exp/zg-validity
Gates pre-registered in docs/02_EXPERIMENTS_AND_RESULTS.md BEFORE this ran.

Subcommands, in the order the gates must run:
    select   draw the G1 strata (frozen to selection.json, seeded)
    g0       instrument trust — corpus z_g must be bit-identical to the zg/ source.
             ABORTS the arm if it fails.
    g2       quantitative diagnostics (confidence + geometric self-consistency)
    g1       render the contact sheets for visual keypoint-plausibility review

CPU-only, NAS I/O bound. No GPU, no model.

Prior-work note (process step 5): the G2 feature set includes *geometric
self-consistency* measures in addition to confidence, following "Detecting Pose
Estimation Failures via Keypoint Self-Consistency" (arXiv 2608.03516), whose central
finding is that geometric self-consistency outperforms confidence-only failure
detection. That method targets RIGID objects (pairwise distances are fixed under
rotation); faces are deformable so it does not transfer literally. The principle
does: do not rely on confidence alone. Added BEFORE any outcome was inspected; the
pre-registered decision rule (G1's 70% thresholds) is unchanged.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # repo root

CORPUS = Path("/mnt/nas-ai-models/training-data/eidolon/hegre_corpus")
SRC = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/stratum/faces")
ZG_SRC = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/zg/faces")
ASSETS = ROOT / "docs/assets/exp/zg-validity"
SELECTION = HERE / "selection.json"

# --- frozen parameters (mirror ../config.yaml; do not edit to taste) ---
FACE_SLICE = slice(23, 91)     # 68 face points from the 133 COCO-WholeBody keypoints
HIGH_MIN = 25.0
CTRL_MIN, CTRL_MAX = 8.0, 12.0
N_HIGH = 60
N_CTRL = 60
SEED = 20260924
THRESHOLDS = (15.0, 25.0)

# 68-point convention, indices within FACE_SLICE
EYE_R = list(range(36, 42))
EYE_L = list(range(42, 48))
MOUTH = list(range(48, 60))


# ---------------------------------------------------------------- helpers
def sample_dirs():
    return sorted(d for d in CORPUS.iterdir() if d.is_dir())


def norm_of(d):
    try:
        v = np.load(d / "z_g.npy")
        return float(np.linalg.norm(v))
    except Exception:
        return None


def source_dir(meta):
    return SRC / meta["persona"] / meta["set"] / meta["image_id"]


def zg_source(meta):
    # zg/ is FLAT: zg/faces/{persona}/{set}/{image_id}.npy  (no sample dir, no zg.npy)
    return ZG_SRC / meta["persona"] / meta["set"] / f"{meta['image_id']}.npy"


# ---------------------------------------------------------------- select
def cmd_select(_args):
    dirs = sample_dirs()
    print(f"scanning {len(dirs)} corpus samples for z_g norms...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        norms = list(ex.map(norm_of, dirs))
    pairs = [(d, n) for d, n in zip(dirs, norms) if n is not None]
    print(f"  {len(pairs)} readable in {time.time() - t0:.0f}s")

    n_all = np.array([n for _, n in pairs])
    dist = {f">{t}": float((n_all > t).mean() * 100) for t in THRESHOLDS}
    print(f"  distribution: p50={np.percentile(n_all,50):.2f} "
          f"p95={np.percentile(n_all,95):.2f} p99={np.percentile(n_all,99):.2f}")
    print(f"  fraction: {dist}")

    rng = random.Random(SEED)
    high = [(d, n) for d, n in pairs if n > HIGH_MIN]
    ctrl = [(d, n) for d, n in pairs if CTRL_MIN <= n <= CTRL_MAX]
    print(f"  eligible: high(>{HIGH_MIN})={len(high)}  control({CTRL_MIN}-{CTRL_MAX})={len(ctrl)}")
    if len(high) < N_HIGH or len(ctrl) < N_CTRL:
        sys.exit(f"FATAL: not enough eligible samples (need {N_HIGH}/{N_CTRL})")

    sel = {
        "seed": SEED,
        "high_min": HIGH_MIN, "ctrl_min": CTRL_MIN, "ctrl_max": CTRL_MAX,
        "n_high": N_HIGH, "n_ctrl": N_CTRL,
        "corpus_norms": {"n": len(pairs),
                         "p50": float(np.percentile(n_all, 50)),
                         "p95": float(np.percentile(n_all, 95)),
                         "p99": float(np.percentile(n_all, 99)),
                         "frac_gt_15": dist[">15.0"], "frac_gt_25": dist[">25.0"]},
        "high": [{"dir": d.name, "norm": n} for d, n in
                 sorted(rng.sample(high, N_HIGH), key=lambda x: -x[1])],
        "ctrl": [{"dir": d.name, "norm": n} for d, n in
                 sorted(rng.sample(ctrl, N_CTRL), key=lambda x: -x[1])],
    }
    SELECTION.write_text(json.dumps(sel, indent=2) + "\n")
    print(f"  wrote {SELECTION.relative_to(ROOT)}  "
          f"(high norms {sel['high'][0]['norm']:.1f}..{sel['high'][-1]['norm']:.1f}; "
          f"ctrl {sel['ctrl'][0]['norm']:.1f}..{sel['ctrl'][-1]['norm']:.1f})")


# ---------------------------------------------------------------- g0
def cmd_g0(_args):
    """Instrument trust. Abort the arm if the corpus is not a faithful copy."""
    dirs = sample_dirs()
    rng = random.Random(SEED)
    probe = rng.sample(dirs, min(400, len(dirs)))
    print(f"G0: corpus z_g vs zg/ source, {len(probe)} random samples")
    bad = missing = 0
    worst = 0.0
    for d in probe:
        meta = json.loads((d / "metadata.json").read_text())
        s = zg_source(meta)
        if not s.exists():
            missing += 1
            continue
        a = np.load(d / "z_g.npy").astype(np.float64)
        b = np.load(s).astype(np.float64)
        diff = float(np.linalg.norm(a - b))
        worst = max(worst, diff)
        if diff > 1e-9:
            bad += 1
    print(f"  mismatches: {bad}   missing source: {missing}   max ||diff||: {worst:.10f}")
    ok = (bad == 0 and missing == 0)
    print(f"  G0 {'PASS' if ok else 'FAIL'} — "
          f"{'corpus is a faithful copy; proceed' if ok else 'ABORT: conclusions would be about the copy, not the encoder'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- g2
def face_metrics(d):
    """Confidence + geometric self-consistency diagnostics for one sample."""
    meta = json.loads((d / "metadata.json").read_text())
    p = source_dir(meta) / "pose.npy"
    out = {"dir": d.name, "norm": norm_of(d), "pose_found": p.exists()}
    if not p.exists():
        return out
    pose = np.load(p).astype(np.float32)
    if pose.shape != (133, 3):
        out["pose_found"] = False
        return out
    xy = pose[FACE_SLICE, :2]
    conf = pose[FACE_SLICE, 2]

    out["conf_mean"] = float(conf.mean())
    out["conf_min"] = float(conf.min())
    out["conf_lt_03"] = int((conf < 0.3).sum())
    out["conf_lt_05"] = int((conf < 0.5).sum())
    out["zero_xy"] = int((xy == 0).all(axis=1).sum())

    # scale-normalised geometry
    span = xy.max(axis=0) - xy.min(axis=0)
    diag = float(np.hypot(*span))
    out["bbox_diag"] = diag
    out["bbox_aspect"] = float(span[0] / span[1]) if span[1] > 1e-9 else np.nan

    er = xy[EYE_R].mean(axis=0)
    el = xy[EYE_L].mean(axis=0)
    mo = xy[MOUTH].mean(axis=0)
    iod = float(np.linalg.norm(el - er))
    out["iod"] = iod
    out["iod_norm"] = iod / diag if diag > 1e-9 else np.nan
    out["eye_mouth_ratio"] = float(np.linalg.norm(mo - (er + el) / 2) / iod) if iod > 1e-9 else np.nan
    out["roll_deg"] = float(np.degrees(np.arctan2((el - er)[1], (el - er)[0])))

    # geometric self-consistency: residual after Procrustes alignment to the
    # canonical template — the unwhitened shape-deviation signal.
    try:
        sys.path.insert(0, str(ROOT / "experiments/geometry_pca"))
        from geometry_pca.gpa import center_and_scale, align_single  # noqa
        enc = np.load(ROOT / "experiments/geometry_pca/output/encoder_production.npz",
                      allow_pickle=True)
        gmean = enc["gpa_mean"]
        c = center_and_scale(xy)
        aligned = align_single(c, gmean)
        out["align_residual"] = float(np.linalg.norm(aligned - gmean))
    except Exception as e:
        out["align_residual"] = np.nan
        out["align_error"] = str(e)[:120]
    return out


def cmd_g2(_args):
    sel = json.loads(SELECTION.read_text())
    names = [s["dir"] for s in sel["high"]] + [s["dir"] for s in sel["ctrl"]]
    groups = ["high"] * len(sel["high"]) + ["ctrl"] * len(sel["ctrl"])
    dirs = [CORPUS / n for n in names]
    print(f"G2: diagnostics on {len(dirs)} selected samples", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(face_metrics, dirs))
    for r, g in zip(rows, groups):
        r["group"] = g
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "g2_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"  wrote {(ASSETS / 'g2_diagnostics.json').relative_to(ROOT)}")

    keys = ["conf_mean", "conf_min", "conf_lt_03", "zero_xy", "iod_norm",
            "eye_mouth_ratio", "roll_deg", "align_residual"]
    print(f"\n  {'metric':<18}{'high (norm>25)':>18}{'control (8-12)':>18}")
    print("  " + "-" * 54)
    for k in keys:
        vals = {g: [r.get(k) for r, gg in zip(rows, groups) if gg == g] for g in ("high", "ctrl")}
        line = f"  {k:<18}"
        for g in ("high", "ctrl"):
            v = np.array([x for x in vals[g] if x is not None and np.isfinite(x)], dtype=float)
            line += f"{v.mean():>18.4f}" if v.size else f"{'n/a':>18}"
        print(line)
    return 0


# ---------------------------------------------------------------- g1
# 68-point skeleton, drawn so plausibility is judgeable at a glance
SKELETON = [
    (list(range(0, 17)), "jaw"),
    (list(range(17, 22)), "brow_r"),
    (list(range(22, 27)), "brow_l"),
    (list(range(27, 31)), "nose_bridge"),
    (list(range(30, 36)), "nose_base"),
    (list(range(36, 42)) + [36], "eye_r"),
    (list(range(42, 48)) + [42], "eye_l"),
    (list(range(48, 60)) + [48], "mouth_out"),
    (list(range(60, 68)) + [60], "mouth_in"),
]
SKEL_COLOR = {"jaw": "#00ffff", "brow_r": "#ffd700", "brow_l": "#ffd700",
              "nose_bridge": "#ff69b4", "nose_base": "#ff69b4",
              "eye_r": "#00ff00", "eye_l": "#00ff00",
              "mouth_out": "#ff4500", "mouth_in": "#ffff00"}


def _denormalize(xy, W, H):
    """[-1,1] normalised pose coords -> pixel coords.

    Convention verified against the writer (stratum-hq, which produces pose.npy):
      src/stratum/pipeline/pose.py:
          x_norm = (2.0 * kpts[:, 0] / bucket_w) - 1.0
          y_norm = (2.0 * kpts[:, 1] / bucket_h) - 1.0
      scripts/visualize_example.py:
          out[:, 0] = (pose[:, 0] + 1.0) * w / 2.0
          out[:, 1] = (pose[:, 1] + 1.0) * h / 2.0

    Both axes are plain image convention (y increases DOWNWARD) — there is NO
    y-flip. An earlier version of this renderer applied py = (1 - (y+1)/2) * H,
    which is a vertical MIRROR and drew the skeleton upside-down. Do not
    reintroduce a flip here.
    """
    px = (xy[:, 0] + 1.0) * W / 2.0
    py = (xy[:, 1] + 1.0) * H / 2.0
    return px, py


def _panel(ax, d, item, draw_skeleton=True):
    meta = json.loads((d / "metadata.json").read_text())
    pix = np.load(d / "pixel.npy").astype(np.float32)
    img = pix.transpose(1, 2, 0)
    img = (img - img.min()) / max(float(np.ptp(img)), 1e-6)
    H, W = img.shape[:2]
    p = source_dir(meta) / "pose.npy"
    if p.exists():
        pose = np.load(p).astype(np.float32)
        xy, conf = pose[FACE_SLICE, :2], pose[FACE_SLICE, 2]
        px, py = _denormalize(xy, W, H)

        # crop around the landmarks so the face fills the panel
        if draw_skeleton:
            mx, my = np.nanmax(px) - np.nanmin(px), np.nanmax(py) - np.nanmin(py)
            x0 = int(max(0, np.nanmin(px) - 0.15 * mx))
            x1 = int(min(W, np.nanmax(px) + 0.15 * mx))
            y0 = int(max(0, np.nanmin(py) - 0.15 * my))
            y1 = int(min(H, np.nanmax(py) + 0.15 * my))
            if x1 - x0 > 8 and y1 - y0 > 8:
                img = img[y0:y1, x0:x1]
                px, py = px - x0, py - y0
        ax.imshow(img)
        if draw_skeleton:
            for idxs, name in SKELETON:
                ax.plot(px[idxs], py[idxs], "-", color=SKEL_COLOR[name],
                        lw=1.1, alpha=0.95)
        good, bad = conf >= 0.3, conf < 0.3
        ax.scatter(px[good], py[good], s=6, c="lime", marker="o",
                   edgecolors="black", linewidths=0.3)
        if bad.any():
            ax.scatter(px[bad], py[bad], s=16, c="red", marker="x")
    ax.set_title(f"{item['norm']:.1f}", fontsize=9)
    ax.axis("off")


def cmd_g1(args):
    """Render contact sheets: DWPose 68-pt skeleton drawn over the pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = json.loads(SELECTION.read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    groups = [args.group] if args.group else ["high", "ctrl"]

    for group in groups:
        items = sel[group]
        ncol, nrow = 10, 6
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 3.0))
        for ax, item in zip(axes.ravel(), items):
            _panel(ax, CORPUS / item["dir"], item)
        fig.suptitle(
            f"G1 {group}: z_g norm {'> 25' if group == 'high' else '8-12'}  "
            f"— 68-pt skeleton; green dot = conf>=0.3, red x = conf<0.3",
            fontsize=13)
        fig.tight_layout()
        out = ASSETS / f"g1_skeleton_{group}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}")
    return 0


def cmd_zoom(args):
    """Large-panel sheet of the top-N samples by norm — for close inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = json.loads(SELECTION.read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    items = sorted(sel[args.group], key=lambda x: -x["norm"])[:args.top]
    ncol = args.cols
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.0, nrow * 5.0))
    axes_flat = np.asarray(axes, dtype=object).ravel()
    for ax, item in zip(axes_flat, items):
        _panel(ax, CORPUS / item["dir"], item)
    for ax in axes_flat[len(items):]:
        ax.axis("off")
    fig.suptitle(f"G1 zoom — top {len(items)} {args.group} by z_g norm "
                 f"(face-cropped, 68-pt skeleton)", fontsize=15)
    fig.tight_layout()
    out = ASSETS / f"g1_zoom_{args.group}{len(items)}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------- review
def _meta_of(d):
    return json.loads((d / "metadata.json").read_text())


def cmd_review(_args):
    """Cross-tab the selected strata (and the whole corpus) against review verdicts.

    The review DB is the human usability judgement. If the corpus carries tainted
    samples, a high-norm stratum drawn from it is measuring the human's rejects,
    not the encoder's failure mode. READ-ONLY: the review UI may be open.
    """
    sys.path.insert(0, str(ROOT))
    from tools.hegre_dataset.config import database_url
    from sqlalchemy import create_engine, text

    sel = json.loads(SELECTION.read_text())
    dirs = sample_dirs()
    print(f"mapping {len(dirs)} corpus samples to review verdicts...", flush=True)
    with ThreadPoolExecutor(max_workers=32) as ex:
        metas = list(ex.map(_meta_of, dirs))
    path_of = {d.name: f"faces/{m['persona']}/{m['set']}/{m['image_id']}.jpg"
               for d, m in zip(dirs, metas)}

    eng = create_engine(database_url())
    status = {}
    with eng.connect() as c:
        paths = sorted(set(path_of.values()))
        for i in range(0, len(paths), 5000):
            chunk = paths[i:i + 5000]
            q = text("SELECT image_path, status FROM images WHERE image_path = ANY(:p)")
            for p, s in c.execute(q, {"p": chunk}):
                status[p] = s
    print(f"  matched {len(status)}/{len(paths)} to review rows\n")

    def tab(rows, label):
        from collections import Counter
        cnt = Counter(status.get(path_of[d.name], "(not in review DB)") for d in rows)
        tot = sum(cnt.values())
        print(f"  {label}  (n={tot})")
        for s, n in cnt.most_common():
            flag = "  <-- TAINTED" if s.startswith("tainted") else ""
            print(f"      {s:<30} {n:>5}  {100*n/tot:5.1f}%{flag}")
        return cnt

    print("=== G1 strata vs review verdict ===")
    hi = tab([CORPUS / s["dir"] for s in sel["high"]], "HIGH  (norm>25)")
    ct = tab([CORPUS / s["dir"] for s in sel["ctrl"]], "CONTROL (norm 8-12)")

    print("\n=== whole corpus vs review verdict ===")
    allc = tab(dirs, "CORPUS (all)")

    out = {"high": dict(hi), "ctrl": dict(ct), "corpus": dict(allc),
           "matched": len(status), "corpus_n": len(dirs)}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "review_crosstab.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {(ASSETS / 'review_crosstab.json').relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("select", cmd_select), ("g0", cmd_g0), ("g2", cmd_g2),
                     ("g1", cmd_g1), ("zoom", cmd_zoom), ("review", cmd_review)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if name == "g1":
            sp.add_argument("--group", choices=["high", "ctrl"], default=None)
        if name == "zoom":
            sp.add_argument("--group", choices=["high", "ctrl"], default="high")
            sp.add_argument("--top", type=int, default=12)
            sp.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
