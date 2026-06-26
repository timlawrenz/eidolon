# Git Workflow & Experiment Isolation

This repository enforces a strict experiment-branching model to prevent code and results from tangling across complex machine learning research phases.

## 1. Branch-to-Experiment Mapping

*   **The `main` Branch**: Reserved for **infrastructure, validated tools, and documentation only** (e.g., pre-training code, validators, unit tests). Highly volatile experimental code, loss functions, and intermediate runs do **NOT** belong here.
*   **The `exp/*` Branches**: Every distinct research hypothesis gets an isolated branch (e.g., `exp/text-to-zg`, `exp/decode-time-solver`).
    *   All messy scripts, local TensorBoard logs, and metric CSVs for that experiment stay frozen on that branch.
    *   Failed experiments are never merged to `main`, but their findings are documented in the ledger on `main`.
    *   Future agents check out fresh `exp/*` branches from `main`.

## 2. Pre-Execution Hygiene (The Branch-Out Rule)

Never begin executing a new plan, task sequence, or extraction on a dirty working tree or an unrelated experiment branch.

**Workflow:**
1. Clean the current state (commit untracked diagnostic scripts, plans, notes).
2. Push the current branch.
3. Check out a fresh `exp/*` branch from `main` for the new work *before* executing the first step.

Attempting to start a new plan without branching tangles history.

## 3. Asset-to-Branch Mapping (Empirical Proof)

All generated assets, raw evaluation logs, plots, and CSVs proving an experiment's result must be saved in `docs/assets/<branch_name>/`. Embed these assets directly into `docs/02_EXPERIMENTS_AND_RESULTS.md` to provide permanent, verifiable empirical proof.

## 4. Scientific Method & Ledger Updates

When starting a new experimental arm:
1. **Formulate:** Define the goal, null-hypothesis ($H_0$), and alternative hypothesis ($H_1$).
2. **Define the Gate:** Pre-register the instrument, metric, and pass/fail threshold.
3. **Write the Ledger:** Add this design to `docs/02_EXPERIMENTS_AND_RESULTS.md` and `docs/03_EXPERIMENT_TREE.md` BEFORE writing implementation scripts.
4. **Execute:** Only after the hypothesis is documented are you permitted to execute the plan.