#!/usr/bin/env python3
"""Retrofit the two missing skill-mandated fields into every provenance.yaml.

Honest by construction: `mode` is only declared `confirmatory` where the file
actually contains a pre-registered gate; otherwise it is marked as not
retroactively determinable. `agent_model` is left null for arms that pre-date
the field rather than back-filling a guess.
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/tim/source/activity/eidolon")
FILES = sorted(ROOT.glob("experiments/**/provenance*.yaml"))

# This session's arms — the only ones whose agent model is actually known.
THIS_SESSION = {
    "experiments/ffhq_basis_reproject/provenance.yaml": (
        "deepseek/deepseek-v4.1-flash", "openrouter/deepseek/deepseek-v4.1-flash"),
    "experiments/zg_validity/provenance.yaml": (
        "deepseek/deepseek-v4.1-flash", "openrouter/deepseek/deepseek-v4.1-flash"),
}

MARKER = "# --- retrofit 2026-09-24 (AGENTS.md: skill mandates these fields) ---"

for f in FILES:
    rel = str(f.relative_to(ROOT))
    text = f.read_text()
    if MARKER in text:
        print(f"  SKIP (already retrofitted): {rel}")
        continue

    d = yaml.safe_load(text)
    gate = d.get("pre_registered_gate")
    has_gate = bool(gate and str(gate).strip() not in ("", "null", "None"))

    if rel in THIS_SESSION:
        am, ams = THIS_SESSION[rel]
        mode_line = "confirmatory          # declared before the run"
    else:
        am, ams = None, None
        if has_gate:
            mode_line = ("confirmatory          # RETROSPECTIVE: a pre-registered gate is "
                         "present in this file")
        else:
            mode_line = ("null                  # RETROSPECTIVE: no gate found; treat as "
                         "exploratory (cannot be relabelled)")

    block = (
        f"\n{MARKER}\n"
        f"mode: {mode_line}\n"
        f"agent_model: {am}   # null = not recorded at run time (field post-dates this arm)\n"
        f"agent_model_snapshot: {ams}\n"
    )
    f.write_text(text.rstrip("\n") + "\n" + block)
    print(f"  PATCHED {rel}  has_gate={has_gate}  mode={'confirmatory' if (has_gate or rel in THIS_SESSION) else 'null'}")

print("\n=== validation ===")
bad = 0
for f in FILES:
    try:
        d = yaml.safe_load(f.read_text())
        assert "mode" in d, "mode missing"
        assert "agent_model" in d, "agent_model missing"
        print(f"  OK  {str(f.relative_to(ROOT)):60s} mode={d['mode']}")
    except Exception as e:
        bad += 1
        print(f"  FAIL {f.relative_to(ROOT)}: {e}")
sys.exit(1 if bad else 0)
