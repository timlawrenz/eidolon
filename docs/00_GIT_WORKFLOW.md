# Git Workflow & Experiment Isolation

This repository enforces a strict experiment-branching model to prevent code and results from tangling across complex machine learning research phases.

## 1. Branch-to-Experiment Mapping

*   **The `main` Branch**: Reserved for **infrastructure, validated tools, and documentation only** (e.g., pre-training code, validators, unit tests). Highly volatile experimental code, loss functions, and intermediate runs do **NOT** belong here.
*   **The `exp/*` Branches**: Every distinct research hypothesis gets an isolated branch (e.g., `exp/text-to-zg`, `exp/decode-time-solver`).
    *   All messy scripts, local TensorBoard logs, and metric CSVs for that experiment stay frozen on that branch.
    *   Failed experiments are never merged to `main`, but their findings are documented in the ledger on `main`.
    *   Future agents check out fresh `exp/*` branches from the **current `main`**.
        `main` must therefore be kept current: if it cannot serve as a base, say so
        explicitly (§2) rather than silently branching from an unrelated `exp/*` branch.
    *   An experiment is identified by its **commit**, not its branch — see §2.

## 2. The Experiment IS a Commit (Evidence Addressability)

A branch is a **moving pointer**. A commit is an **immutable snapshot**. An experiment
is therefore identified by a **commit**, never by a branch: *the way the code was at
commit `abc` is what we consider the experiment.* Branches may move, be renamed, or be
deleted; they are only where the commit currently happens to sit.

**The rule:**

1. **Name the commit.** Every experiment entry in `docs/02_EXPERIMENTS_AND_RESULTS.md`
   carries `**Commit:** <full 40-hex SHA>` — the tree at that SHA *is* the experiment.
   Short SHAs and branch names are not acceptable (branches move; short SHAs are
   ambiguous).
2. **Capture it automatically.** The commit is recorded by the **run script**, not typed
   by hand — the run writes `git rev-parse HEAD` plus a clean-tree check into its own
   metrics JSON. A hand-transcribed SHA is an unverified claim; a machine-captured one
   is evidence. Record the tree state too: a dirty tree means the SHA is *not* the
   experiment, and that must fail loudly rather than be silently tolerated.
3. **Keep it reachable.** A commit survives only while reachable; delete the branch and
   it is eventually garbage-collected. Every experiment commit is therefore **tagged**,
   which pins it permanently and independently of any branch. Existing convention
   (`phase2-zd-concluded`, `phase2b-za-concluded`) already does this for concluded arms;
   the rule generalises it to all arms.
4. **Resolve citations against the commit.** `git show <SHA>:<path>` reads any cited
   artifact branch-independently. A ledger citation is resolved against **its own
   entry's commit** — not against `HEAD`, and not against `main`.
5. **The commit must hold the code AND the evidence.** The experiment commit is the
   commit at which the run code, the **evidence artifact**, and the results entry are
   *all present* — normally the commit that closes the arm. A commit carrying the code
   but not the evidence verifies nothing: it reproduces the run without the thing the
   run produced. This is the stricter reading, and it is the one enforced: an
   evidence-less commit was rejected by the lint during the rule's own introduction
   (the run-code commit `ca1b49b` lacked the `fisher_metrics.json` that later landed
   in `29aba37`). Committing evidence *after* the code is the common way to violate
   this without noticing.

**What this buys:** an arm's code does *not* need to live on `main` for its numbers to be
verifiable. `main` stays reserved for infrastructure, validated tools, and documentation
(§1), while every experiment number remains permanently addressable by SHA. This removes
the false choice between "merge everything to `main`" and "citations only resolve on the
branch where they happened".

**Two failure modes this rule exists to catch:**

* **Citation without a commit** — unverifiable by construction; must fail the governance
  lint rather than pass by omission.
* **Cited commit orphaned** (branch deleted, not tagged) — the number is still *true*
  but no longer *checkable*. Tagging is what makes a result durable evidence.

**Stale SHAs on live arms.** An arm's SHA is fixed at the moment of its run; later
doc-only commits do not change it. Cite the commit that contains the **run code**, which
is normally the commit that last touched the arm's `src/` before the run — not the
branch tip, which will drift.

## 3. Pre-Execution Hygiene (The Branch-Out Rule)

Never begin executing a new plan, task sequence, or extraction on a dirty working tree or an unrelated experiment branch.

**Workflow:**
1. Clean the current state (commit untracked diagnostic scripts, plans, notes).
2. Push the current branch.
3. Check out a fresh `exp/*` branch from `main` for the new work *before* executing the first step.

Attempting to start a new plan without branching tangles history.

## 4. Asset-to-Branch Mapping (Empirical Proof)

All generated assets, raw evaluation logs, plots, and CSVs proving an experiment's result must be saved in `docs/assets/<branch_name>/`. Embed these assets directly into `docs/02_EXPERIMENTS_AND_RESULTS.md` to provide permanent, verifiable empirical proof.

## 5. Scientific Method & Ledger Updates

When starting a new experimental arm:
1. **Formulate:** Define the goal, null-hypothesis ($H_0$), and alternative hypothesis ($H_1$).
2. **Define the Gate:** Pre-register the instrument, metric, and pass/fail threshold.
3. **Write the Ledger:** Add this design to `docs/02_EXPERIMENTS_AND_RESULTS.md` and `docs/03_EXPERIMENT_TREE.md` BEFORE writing implementation scripts.
4. **Execute:** Only after the hypothesis is documented are you permitted to execute the plan.