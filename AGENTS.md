# Project Eidolon — Agent Instructions

This file exists for AI agents (Hermes, Claude, Codex, etc.) working on this
project. Read it FIRST — before reading any other file, before running any
commands, before making any changes.

## Start here (mandatory — read in this order)

1. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** — The living pointer. Current
   phase, headline results, and the single next action. This tells you whether
   the project is active, between phases, or blocked. **Read this before
   anything else.**

2. **[docs/experiment-structure.md](docs/experiment-structure.md)** — The
   governance contract. This is the single most important document for
   understanding how experiments are run and recorded. It contains:
   - The directory layout and naming conventions
   - The provenance.yaml and config.yaml format
   - The pre-registered gate system
   - The adversarial pass checklist (mandatory before any PASS verdict)
   - The project verdict vocabulary (GO/PIVOT/PARK/KILL)
   - The **Process for Agents** section (16 numbered steps — follow them in order,
     but the *skill* is authoritative if this copy lags; see below)

3. **[docs/03_EXPERIMENT_TREE.md](docs/03_EXPERIMENT_TREE.md)** — The living
   workstream map. Check this BEFORE starting any new experiment to see if it's
   already `[CONCLUDED]` or `[ACTIVE]`. Never re-run a concluded experiment
   without explicit user direction.

4. **[docs/02_EXPERIMENTS_AND_RESULTS.md](docs/02_EXPERIMENTS_AND_RESULTS.md)** —
   The permanent ledger. Every experiment has a dated entry with pre-registered
   gates, empirical evidence, and verdicts. Check this before proposing any
   hypothesis — the answer may already be documented as a negative result.

## Governance: the skill is the source of truth

This repo's governance docs are a **tailored instance** of the
`scientific-experiment-structure` skill. **Load the skill before running or
recording any experiment:**

```
skill_view(name='scientific-experiment-structure')
```

**When this repo's docs and the skill disagree, the skill wins on process.** The
project docs win only on project-specific facts (paths, hostnames, naming,
hardware). `docs/experiment-structure.md` is a *copy* and **can lag the skill** —
it already has. Never treat a project doc as the last word on process; when in
doubt, re-read the skill.

### Known drift in `docs/experiment-structure.md` (as of 2026-09-24)

The project copy is missing rules the skill requires. Treat the skill as
authoritative for each of these:

| Rule | Skill requirement | Project copy |
|---|---|---|
| Arm kind | `mode: confirmatory\|exploratory` in `provenance.yaml`, **before the first run**. Only a confirmatory arm may write PASS/FAIL | **absent** |
| Agent provenance | `agent_model` + `agent_model_snapshot` — this project is agent-assisted | **absent** |
| Adversarial pass | **6 boxes** (incl. "headline number traced to an exact artifact" and "every flaw found is FIXED or explicitly gated-not-fixed") | 4 boxes |
| Peeking | "Peeked = exploratory, period" — a gate locked after seeing the outcome cannot be relabelled confirmatory | not stated |
| Feasibility | Feasibility mode is opt-in/opt-out by the user alone; feasibility results are **not evidence** | not stated |
| Step count | 16 numbered steps in the Process for Agents | says 14 |

## Critical rules (break these and you will waste real compute)

- **Never re-run a `[CONCLUDED — FAIL]` or KILLed experiment** without explicit
  user direction AND a new hypothesis. The ledger exists to prevent this.
- **Never skip the adversarial pass** before writing a PASS verdict. A measured
  PASS that was actually a measurement bug is the most expensive failure mode in
  this project. See the Phase 2b (z_a) overturn as a real example.
- **Always state pre-registered gates BEFORE seeing results.** Write the gate in
  the ledger, then run the experiment, then fill in the evidence. Never the
  reverse.
- **Always verify identity test sets visually.** Contamination (name collisions,
  couple-shoot faces, seg-collapse) has nearly killed valid results. See the
  Phase 1-R contamination near-miss.
- **Config keys that look right but aren't consumed by code produce silently
  invalid experiments.** grep-trace every config key through the codebase.
- **Declare `mode: confirmatory | exploratory` in `provenance.yaml` before the
  first run.** Only a *confirmatory* arm may write PASS/FAIL in the ledger. An
  arm that peeked at its outcome before locking its gate is **exploratory** and
  cannot be relabelled.
- **Evidence must live in the repo, not in a scratch directory.** The agent
  scratch dir (`~/.hermes/profiles/eidolon/cache/scratch/`) is **pruned after
  24h idle**. Any script or number a governance doc cites must be committed —
  put it in the arm's `src/` or in `docs/assets/<branch>/`. A ledger number
  whose producing script no longer exists is not evidence.
- **Put experiment assets in `docs/assets/<branch_name>/`** (see
  `docs/00_GIT_WORKFLOW.md` rule 3) — plots, metric JSON, contact sheets, raw
  eval logs. The adversarial pass requires an *artifact* for "extremes
  inspected"; cite the committed path, not a scratch path.
- **Branch from a clean tree on the correct base before executing anything new**
  (`docs/00_GIT_WORKFLOW.md` rule 2). If the correct base lacks a dependency the
  arm needs, say so explicitly rather than silently branching from an unrelated
  `exp/*` branch.
- **A KILL requires a `DISCONTINUATION_NOTICE.md`.** The tree marking something
  "dead" is not a tombstone — the notice must state the *structural* reason it
  cannot work, or a future agent will re-attempt it.
- **Arm-specific code goes in `experiments/{arm}/src/`,** not in shared
  `scripts/` or `production/`. Shared tools are fine in `tools/`; one-off
  experiment scripts are not.

## Project structure

```
eidolon/
├── PROJECT_STATUS.md              ← READ FIRST
├── AGENTS.md                      ← This file
├── README.md                      ← Human-facing overview
├── docs/                          ← Governance (all in git)
│   ├── experiment-structure.md    ← Rules for running experiments (⚠️ may lag the skill)
│   ├── 00_GIT_WORKFLOW.md         ← Branch-to-experiment mapping
│   ├── 01_VISION_AND_ARCHITECTURE.md  ← Canonical architecture
│   ├── 02_EXPERIMENTS_AND_RESULTS.md  ← Permanent ledger
│   ├── 03_EXPERIMENT_TREE.md      ← Living workstream map
│   ├── briefings/                 ← Cross-project handoff documents
│   └── assets/<branch_name>/      ← Committed evidence: plots, metric JSON, contact sheets
├── experiments/                   ← Experiment arms (code + provenance)
│   ├── geometry_pca/              ← Phases 1–4 (concluded, sub-arm split)
│   │   ├── provenance_zg_posenorm.yaml, config_zg_posenorm.yaml
│   │   ├── provenance_zd_depth.yaml, config_zd_depth.yaml
│   │   ├── provenance_za_normals.yaml, config_za_normals.yaml
│   │   ├── provenance_dino_bridge.yaml, config_dino_bridge.yaml
│   │   └── provenance_dino_patches.yaml, config_dino_patches.yaml
│   ├── sapiens2_keypoints/        ← Sapiens2 study (concluded)
│   │   ├── provenance.yaml, config.yaml
│   │   └── README.md
│   ├── ffhq_basis_reproject/      ← FFHQ identity reprojection (concluded — PASS)
│   └── zg_validity/               ← z_g validity threshold (pre-registered)
├── tools/hegre_dataset/           ← Shared dataset infrastructure
├── tests/                         ← Tests for shared tools
└── scripts/                       ← Pipeline and migration scripts
```

## Branch structure

- **`main`** — Finalized documentation, infrastructure, and validated tools.
  No active experimental code lives here.
- **`exp/*`** branches — Each experiment arm lives on its own branch. Check
  the experiment tree (`docs/03_EXPERIMENT_TREE.md`) for the mapping.
- Read `docs/00_GIT_WORKFLOW.md` before checking out or creating branches.

## Code understanding

The project is indexed by the codebase-memory-mcp knowledge graph. For
structural questions (where is X defined, what calls X, what's the
architecture), use the graph tools. See the `codebase-memory-mcp` skill for
full workflow reference.

## Training environment

| Resource | Detail |
|---|---|
| Training host | game |
| Training GPU | RTX 4090 (24GB) |
| NAS (experiment data) | `/mnt/nas-ai-models/training-data/eidolon/` |
| Hegre dataset | `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/` |
| Strix Halo box | Data feeder only (not for training) |

## Verdicts at a glance

| Verdict | Meaning |
|---|---|
| **GO** | Hypothesis held; continue / scale / productionize |
| **PIVOT** | Core idea partially works; redirect |
| **PARK** | Inconclusive, blocked on external input |
| **KILL** | Hypothesis disproven. Requires DISCONTINUATION_NOTICE.md |

## Current state (see PROJECT_STATUS.md for details)

- **Phase 5b concluded** — Poser retrieval spike (GT-LDA ceiling R@1=0.8538 after the
  2026-07-23 basis refit; was 0.842 on the pre-refit basis)
- **Conditioning stack settled** — DINOv3 patches (identity) + z_g/DWPose (pose) + Sapiens2 (shape)
- **Dead partitions** — z_d (depth), z_a (normals), DINO bridge (all KILLed;
  tombstones written — `docs/DISCONTINUATION_NOTICE_{zd_depth,za_normals,dino_bridge}.md`)
- **Data integrity (2026-09-24)** — identity streams all on one basis
  (`120e1c5a1dc4f423`, L2-normalized, stamped); `z_g` high-norm tail still open
  (`exp/zg-validity`, gate approved, not yet run)
- **Next** — Phase 5: DiT Fusion Stack (2-stream decoupled cross-attention)
- **No active training runs**
