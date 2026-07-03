#!/usr/bin/env python3
"""Phase 1: Inventory audit for hegre-faces/v1 corpus readiness.

Queries the hegre-faces/v1 review.db and produces a statistical inventory.
File existence checks use random sampling (fast on CIFS NAS) rather than
checking every file.
"""

import sqlite3
import json
import sys
import random
from pathlib import Path
from collections import defaultdict

DB_PATH = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
DATA_ROOT = Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1")
SAMPLE_PER_PERSONA = 10   # check up to N files per persona for existence
SEED = 42


def main():
    random.seed(SEED)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&nolock=1", uri=True)
    c = conn.cursor()

    # ── Contamination set ────────────────────────────────────────────
    c.execute("SELECT DISTINCT persona_id FROM images WHERE status='tainted:contamination'")
    contaminated = {r[0] for r in c.fetchall()}

    # ── Fetch all approved (non-bad-geometry) ─────────────────────────
    c.execute("""
        SELECT i.persona_id, p.name, i.image_path 
        FROM images i
        JOIN personas p ON i.persona_id = p.id
        WHERE i.status = 'approved'
        AND i.persona_id NOT IN (
            SELECT DISTINCT persona_id FROM images WHERE status = 'tainted:contamination'
        )
        ORDER BY p.name
    """)
    rows = c.fetchall()
    conn.close()

    # ── Per-persona aggregation ───────────────────────────────────────
    personas = defaultdict(lambda: {
        "name": "",
        "approved": 0,
        "paths": [],
        "contaminated": False,
    })

    for persona_id, persona_name, image_path in rows:
        p = personas[persona_id]
        p["name"] = persona_name
        p["approved"] += 1
        p["contaminated"] = persona_id in contaminated
        p["paths"].append(image_path)

    # ── Sample-based file existence check ─────────────────────────────
    file_stats = {
        "pixel": {"found": 0, "checked": 0, "missing_examples": []},
        "z_g": {"found": 0, "checked": 0, "missing_examples": []},
        "auraface_raw": {"found": 0, "checked": 0, "missing_examples": []},
        "auraface_lda": {"found": 0, "checked": 0, "missing_examples": []},
    }

    for pid, info in personas.items():
        sample_paths = random.sample(info["paths"], min(SAMPLE_PER_PERSONA, len(info["paths"])))
        for img_path in sample_paths:
            rel_base = Path(img_path).with_suffix('')

            for key, subdir, suffix in [
                ("pixel", "", ".jpg"),
                ("z_g", "zg/", ".npy"),
                ("auraface_raw", "auraface/", ".npy"),
                ("auraface_lda", "lda/", ".npy"),
            ]:
                fpath = DATA_ROOT / subdir / f"{rel_base}{suffix}"
                file_stats[key]["checked"] += 1
                if fpath.exists():
                    file_stats[key]["found"] += 1
                elif len(file_stats[key]["missing_examples"]) < 5:
                    file_stats[key]["missing_examples"].append(str(fpath))

    # ── Per-persona averages check ────────────────────────────────────
    avg_dir = DATA_ROOT / "averages"
    avg_stats = {"zg": 0, "auraface": 0, "lda": 0}
    for pid, info in personas.items():
        name = info["name"]
        if (avg_dir / f"{name}.zg.npy").exists():
            avg_stats["zg"] += 1
        if (avg_dir / f"{name}.auraface.npy").exists():
            avg_stats["auraface"] += 1
        if (avg_dir / f"{name}.lda.npy").exists():
            avg_stats["lda"] += 1

    # ── Tier categorization ───────────────────────────────────────────
    def tier(info):
        n = info["approved"]
        if n >= 20:
            return "full"
        elif n >= 5:
            return "enough"
        elif n >= 1:
            return "some"
        return "none"

    tiers = defaultdict(list)
    for pid, info in personas.items():
        t = tier(info)
        tiers[t].append((pid, info))

    # ── Print report ──────────────────────────────────────────────────
    total_personas = len(personas)
    total_approved = sum(info["approved"] for _, info in personas.items())

    print("=" * 70)
    print("HEGRE CORPUS INVENTORY (sample-based)")
    print("=" * 70)
    print(f"Contamination-free personas:    {total_personas}")
    print(f"Total approved images:          {total_approved}")
    print(f"Samples checked per persona:    up to {SAMPLE_PER_PERSONA}")
    print()

    print("─" * 70)
    print("TIER BREAKDOWN (by approved count)")
    print("─" * 70)
    for t in ["full", "enough", "some", "none"]:
        items = tiers.get(t, [])
        imgs = sum(info["approved"] for _, info in items)
        print(f"  {t:8s}: {len(items):4d} personas, {imgs:6d} images "
              f"(min {min((info['approved'] for _, info in items), default=0)}, "
              f"max {max((info['approved'] for _, info in items), default=0)})")

    print()
    print("─" * 70)
    print("FILE EXISTENCE (sampled)")
    print("─" * 70)
    for key, label in [
        ("pixel", "Face crops (pixels)"),
        ("z_g", "z_g vectors"),
        ("auraface_raw", "AuraFace raw"),
        ("auraface_lda", "AuraFace LDA"),
    ]:
        s = file_stats[key]
        pct = s["found"] / s["checked"] * 100 if s["checked"] > 0 else 0
        print(f"  {label:25s}: {s['found']:6d} / {s['checked']:6d} ({pct:5.1f}%)")
        if s["missing_examples"]:
            print(f"    Sample missing: {s['missing_examples'][0]}")

    print()
    print("─" * 70)
    print("PER-PERSONA AVERAGES")
    print("─" * 70)
    for key, label in [
        ("zg", "z_g average"),
        ("auraface", "AuraFace average"),
        ("lda", "AuraFace-LDA average"),
    ]:
        pct = avg_stats[key] / total_personas * 100 if total_personas > 0 else 0
        print(f"  {label:25s}: {avg_stats[key]:4d} / {total_personas} ({pct:5.1f}%)")

    print()
    print("─" * 70)
    print("TOP 10 PERSONAS")
    print("─" * 70)
    sorted_ppl = sorted(personas.items(), key=lambda x: x[1]["approved"], reverse=True)
    for pid, info in sorted_ppl[:10]:
        t = tier(info)
        print(f"  {info['name']:20s}  {info['approved']:5d} approved  tier={t}")

    # ── Save summary JSON ─────────────────────────────────────────────
    output = {
        "total_personas_clean": total_personas,
        "total_approved": total_approved,
        "tiers": {t: len(items) for t, items in tiers.items()},
        "tier_images": {t: sum(info["approved"] for _, info in items) for t, items in tiers.items()},
        "file_existence_sample": {
            key: {"found": s["found"], "checked": s["checked"],
                  "pct": round(s["found"] / s["checked"] * 100, 1) if s["checked"] > 0 else 0}
            for key, s in file_stats.items()
        },
        "averages": avg_stats,
        "data_root": str(DATA_ROOT),
        "paths": {
            "pixel": "faces/{persona}/{set}/{img}.jpg",
            "z_g": "zg/faces/{persona}/{set}/{img}.npy",
            "auraface_raw": "auraface/faces/{persona}/{set}/{img}.npy",
            "auraface_lda": "lda/faces/{persona}/{set}/{img}.npy",
            "average_lda": "averages/{persona}.lda.npy",
        },
    }

    out_path = Path(__file__).parent / "hegre_corpus_inventory.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
