# Experiment Directory Structure — Eidolon

**⚠️ This document is a tailored copy of the project-independent
`scientific-experiment-structure` skill, and it can lag it.** The skill is the
**source of truth for process**. This document is authoritative only for
**project-specific facts** — NAS paths, hostnames, hardware, Eidolon conventions,
and the project's own naming scheme.

Load the skill before running or recording an experiment:

```
skill_view(name='scientific-experiment-structure')
```

**If the two disagree, the skill wins.** Drift previously found in this copy is
recorded in `AGENTS.md` — that list is an *example* of the kind of drift that
happens here, not a complete inventory. Nothing in this repo detects a skill-side
change, so re-read the skill rather than trusting this copy to be current.

## Core Principles

1. **Honest science.** Pre-register gates BEFORE seeing results. Negative results
   are recorded permanently — never let a dead end be re-run.
2. **Immutable records.** Config is frozen at run start. The copy in
   `runs/{timestamp}/` is permanent — never edit it.
3. **One arm = one directory.** Each experimental condition gets its own
   `experiments/{slug}/` directory with a single hypothesis.
4. **Separate data from governance.** Experiment data (checkpoints, cached
   features) lives on NAS (`/mnt/nas-ai-models/training-data/eidolon/`).
   Governance documents (structure, tree, ledger, this file) live in git.

## Directory Layout

```
eidolon/
├── experiments/                         # Code + governance in git
│   └── {arm-slug}/                      # Descriptive lowercase-hyphen slug
│       ├── README.md                    # Hypothesis, rationale, expected outcome
│       ├── config.yaml                  # Frozen canonical config
│       ├── provenance.yaml              # Machine-readable record
│       ├── DISCONTINUATION_NOTICE.md    # Only if KILLed (see Verdicts below)
│       └── runs/                        # Data — symlink to NAS, or local for
│           └── {YYYY-MM-DD_HHMM}/       #   small artifacts
│               ├── config.yaml          # Immutable copy from run start
│               ├── metadata.json        # Git commit, command, host/GPU, dirty flag
│               ├── training_log.jsonl   # Step-level metrics (optional)
│               ├── tensorboard/         # TensorBoard event files
│               ├── checkpoints/         # epoch_*.pt, best.pt
│               ├── nohup.log            # Raw stdout/stderr
│               └── figures/             # Generated plots
├── docs/                                # Governance (all in git)
│   ├── 00_GIT_WORKFLOW.md               # Branch-to-experiment mapping rules
│   ├── 01_VISION_AND_ARCHITECTURE.md    # Canonical architecture reference
│   ├── 01_LITERATURE_SYNTHESIS.md       # Literature survey + cross-references
│   ├── 02_EXPERIMENTS_AND_RESULTS.md    # Permanent ledger (dated findings)
│   ├── 03_EXPERIMENT_TREE.md            # Living workstream map
│   ├── 05_PIPELINE_REFERENCE.md         # Pipeline reference
│   ├── experiment-structure.md          # This file
│   └── PROJECT_STATUS.md                # Living orientation pointer
└── src/ / tools/ / tests/               # Shared infrastructure
```

> **Note on naming:** The docs are numbered (`00_`, `01_`, `02_`, etc.) to
> signal reading order. This is an intentional divergence from the generic skill
> convention. When documents are referenced by role (e.g., "the ledger"),
> they mean the numbered file: ledger = `02_EXPERIMENTS_AND_RESULTS.md`, tree =
> `03_EXPERIMENT_TREE.md`.

### NAS paths

The NAS root for Eidolon experiment data (checkpoints, feature caches, large
artifacts) is:

```
/mnt/nas-ai-models/training-data/eidolon/
```

Symlink `runs/` inside each arm to the corresponding NAS directory if the
artifacts are large. Small artifacts (metadata.json, config copies, figures
under 10MB) can live directly in git.

## provenance.yaml Template

Every arm must have a `provenance.yaml` before the first run:

```yaml
arm: ce-baseline                    # matches directory name
mode: confirmatory                  # confirmatory | exploratory — REQUIRED, see below
hypothesis: >
  One sentence stating what this arm tests.
  MUST be falsifiable: state the outcome that would prove it WRONG.
falsified_if: >
  The specific condition that disproves the hypothesis.
pre_registered_gate: "PASS if ..."  # Stated BEFORE seeing results
differs_from: zg-baseline           # or null if baseline
diff_summary: |
  - loss_type: ce                   # bullet list of what changed vs baseline
  - class_weights: none
git_commit: a1b2c3d                 # filled at run start
git_dirty: false
training_host: game                 # hostname
training_gpu: RTX 4090
agent_model: null                   # LLM driving the research (agent-assisted arms)
agent_model_snapshot: null          # pinned provider/model snapshot
training_epochs: 50
data_snapshot: "1475 images, 18568 label points"
backbone: sapiens2_0.4b (frozen)
notes: ""
```

### `mode:` — confirmatory vs exploratory (the anti-HARKing rule)

**Set `mode:` before the first run.** Every arm is exactly one of two kinds.

- **Confirmatory** — has a pre-registered gate (or a falsifiable hypothesis), run to
  produce a verdict that counts as evidence. **Only a confirmatory arm may write
  PASS/FAIL in the ledger.**
- **Exploratory** — probing for signal with no pre-registered gate: sweeps,
  calibration surveys, "what happens if…" runs, feasibility runs. Its outputs are
  **leads, not conclusions.** An interesting exploratory result must be re-registered
  as a fresh confirmatory arm, on data not yet used, before it counts as evidence.

**Peeked = exploratory, period.** If you looked at the outcome before locking the
gate, that analysis is exploratory — the data has already shaped what you
"predicted". You cannot relabel it confirmatory. Write the hypothesis down post hoc
and pre-register a fresh test, or record the arm as exploratory. **Never promote a
peeked result to PASS.**

**Decision rules must be result-independent.** A cutoff moved to where it passed, a
covariate added after seeing the number, a subgroup chosen because it was
significant — all of these make the result exploratory no matter how it is labelled.
Fix the rule before the result exists.

### Feasibility before registration

Pre-registering a gate over a configuration that has never run is writing pass/fail
rules for behaviour you have not seen. When "can this even run at scale?" is the open
question, resolve feasibility first, then register the study on **measured** numbers.

- **Nothing is feasible until it has run.** An estimate is not a measurement.
- **Nothing is bounded until the bound has fired.** A kill criterion never observed
  firing is a hope, not a limit.

**Feasibility mode is opt-in and opt-out by the user alone.** It defers — never
cancels — the plan and the pre-registration. **Feasibility results are not evidence.**
Do not enter it to dodge pre-registration, and do not let a promising exploratory
result silently become "the study."

## config.yaml Rules

1. **Config is frozen at run start.** The canonical `config.yaml` lives in
   `experiments/{slug}/`. The training script copies it into
   `runs/{timestamp}/config.yaml` — that copy is immutable.
2. **Never edit a run's config.** If you need a variant, create a new arm.
3. **Dead config keys are a known pitfall.** Always verify every key is
   actually consumed by the code (`grep`-trace each key through the codebase).
   PyTorch `CrossEntropyLoss` needs an explicit `weight` tensor — a
   `class_weights` array in config that isn't wired through is silently ignored.

### Script-driven experiments

Many Eidolon experiments are script-driven (numbered Python scripts) rather
than config-file-driven. For these arms:

- Extract the key parameters into `config.yaml` even if the scripts hardcode
  them. The config is a *record*, not necessarily the runtime source of truth.
- Note in `provenance.yaml` which script(s) constitute the run.
- Multiple scripts in one arm that test different hypotheses → split into
  separate arms (see Phase 1 vs Phase 2 vs Phase 3 as precedent).

### Sub-arm conventions (concluded work in shared directories)

The `experiments/geometry_pca/` directory contains concluded work spanning
multiple phases. Rather than create empty sub-directories just for provenance,
these sub-arms use **suffixed filenames** within the shared directory:

```
experiments/geometry_pca/
├── README.md
├── provenance_zg_posenorm.yaml     # Phase 1-R: 3D frontalized geometry encoder
├── config_zg_posenorm.yaml
├── provenance_zd_depth.yaml        # Phase 2:   depth partition (KILLed)
├── config_zd_depth.yaml
├── provenance_za_normals.yaml      # Phase 2b:  surface normals (KILLed)
├── config_za_normals.yaml
├── provenance_dino_bridge.yaml     # Phase 3:   DINO→slider bridge (KILLed)
├── config_dino_bridge.yaml
├── provenance_dino_patches.yaml    # Phase 4:   masked patch token identity
├── config_dino_patches.yaml
├── scripts/                        # Numbered scripts constituting the runs
└── ...
```

The arm-slug embedded in the filename (`zg_posenorm`, `zd_depth`, etc.) is
the canonical arm identifier used in the tree and ledger. All rules for
provenance.yaml and config.yaml apply identically — only the file naming
differs.

**For NEW work going forward:** create a top-level `experiments/{arm-slug}/`
directory following the standard convention. The suffixed pattern is a
retrofit for concluded work and should not be used for new experiments.

## Creating a New Experiment Arm

```bash
ARM=my-experiment
mkdir -p experiments/$ARM/runs

# 1. README — hypothesis and expected outcome
cat > experiments/$ARM/README.md << EOF
# $ARM
**Hypothesis:** ...
**Differs from baseline:** ...
**Expected outcome:** ...
EOF

# 2. Config — canonical parameters
cp experiments/baseline-arm/config.yaml experiments/$ARM/config.yaml
# then edit

# 3. Provenance
cat > experiments/$ARM/provenance.yaml << EOF
arm: $ARM
hypothesis: > ...
falsified_if: > ...
pre_registered_gate: "PASS if ..."
differs_from: baseline-arm
diff_summary: |
  - ...
git_commit: $(git rev-parse HEAD)
git_dirty: $(test -z "$(git status --porcelain)" && echo false || echo true)
training_host: $(hostname)
notes: ""
EOF

# 4. Register in the tree
# Add entry to docs/03_EXPERIMENT_TREE.md under Active or TBD

# 5. Register in the ledger
# Add a pre-registered gate entry to docs/02_EXPERIMENTS_AND_RESULTS.md
# BEFORE running. The gate is stated before results — this is non-negotiable.
```

## Pre-Registered Gates

Before running any experiment, state the pass/fail criteria in the ledger
(`02_EXPERIMENTS_AND_RESULTS.md`). Format:

```markdown
**Pre-registered gate (stated BEFORE results):**
> PASS if val_acc ≥ 0.94 AND val_loss does not diverge.
> FAIL if val_acc < 0.92 OR val_loss increases monotonically after epoch 10.
```

The gate must be **falsifiable** — it must state the condition that disproves
the hypothesis. If you can't write a gate, the hypothesis is too vague to run.

## Adversarial Pass (mandatory before any PASS verdict)

**Never write `PASS` in the ledger until you have tried to prove the result is a
lie.** A green metric is a hypothesis, not a conclusion. The most expensive
failure mode is a *believed* result that was actually a measurement bug. Run
this gate on every candidate PASS. If any answer is "no" or "unsure,"
the verdict is `PENDING`, not `PASS`.

**Gate zero — compute the null before the first run.** What would a dummy achieve?
Always predict the majority class; always guess the mean; always output zero. If the
null is close to your threshold, the metric cannot discriminate and the experiment
cannot fail in a detectable way.

**The null must match the feature type.** For a **learned mapping**, the null is a
**random projection of the same input** — the question is whether training added
anything over the input. For a **raw geometric feature**, a random-projection null is
*invalid*: Johnson–Lindenstrauss preserves cosine, so projection ≈ feature and the
null is artificially high. Use **label-shuffle** there. Applying the wrong null in
either direction yields a confident wrong answer — see the DINOv3 bridge tombstone.

1. **Is the metric's own code tested?** The validator/scorer/harness must have
   unit tests. An untested validator produces confident, precise, wrong numbers.
   → Commit: ______
2. **Has the metric definition stayed stable across the runs you're comparing?**
   If the validator changed between arm A and arm B, their numbers are not
   comparable. Re-run the baseline under the current definition.
   → Metric version: ______
3. **Is the result reproducible?** Re-run the winning config (or eval) once with
   a different seed or a fresh process. A PASS that only appears once is a
   coincidence.
   → Reproduction run: ______
4. **Do the extremes and edge cases look right?** Pull the top / bottom /
   dead-center predictions and eyeball them. Confirm the win isn't an averaging
   artifact.
   → Artifact: ______
5. **Is the headline number traced to an exact artifact?** An exact log line, a
   checkpoint metric, a tensorboard value, a committed JSON. A number recalled from
   memory or paraphrased from a console session is **not** evidence.
   → Path: ______
6. **Is every flaw found in review FIXED, or explicitly gated-not-fixed?**
   Diagnosing a flaw is not a verdict. A known-broken result shipped with its flaw
   merely *acknowledged* is a `FAIL`, not a `PASS-with-caveat`.
   → Fix commit, or the explicit decision not to fix + reason: ______

```markdown
**Adversarial pass (fill BEFORE writing the verdict):**
- [ ] Null computed before the first run — value: ______
- [ ] Metric tested — commit: ______
- [ ] Metric definition stable — version: ______
- [ ] Result reproduced — run: ______
- [ ] Extremes inspected — artifact: ______
- [ ] Headline number traced to an exact artifact — path: ______
- [ ] Every flaw found is FIXED or explicitly gated-not-fixed — commit / decision: ______
Verdict: PASS / FAIL / PENDING  (PENDING if any box is unchecked)
```

> **Real example from this project:** Phase 2b (normals, z_a) was logged as PASS
> with ΔAUC +0.028, then overturned when the editorial-keypoint z_g baseline
> (0.540) was discovered to be a resolution artifact. The real baseline (0.688)
> showed z_a *subtracts* −0.039. The adversarial pass is not theoretical.

### Evidence must live in the repo, not in a scratch directory

Any number a governance doc cites must have its **producing script committed** — in
the arm's `src/` or in `docs/assets/<branch>/`. Agent scratch/temp directories are
pruned (after 24h idle), so a ledger citing `scratch/foo.py` is citing code that will
not exist when the number is next questioned. This is box 5 applied to the evidence
itself.

### Retrofitting fields into existing provenance files

**Do not back-fill a guess.** Declare `mode: confirmatory` only where a pre-registered
gate is actually present in the file; leave `agent_model: null` with a comment for arms
that pre-date the field. A retrospective marker recording that the value is retroactive
is more honest than a plausible fabrication — and the fabrication is undetectable later.

## Three-Document System

### 1. `03_EXPERIMENT_TREE.md` — Living Workstream Map

A shallow tree of active, planned, and concluded work. Status tags: `[ACTIVE]`,
`[CONCLUDED]`, `[TBD]`. Each entry links to the experiment arm. Brief summaries
with key numbers — evidence lives in the ledger.

### 2. `02_EXPERIMENTS_AND_RESULTS.md` — Permanent Ledger

Every experiment gets a dated entry with: Goal, Pre-registered gate (stated
BEFORE results), Empirical Evidence, Adversarial Pass checklist, Verdict
(PASS/FAIL/CONCLUDED). Negative results are honored permanently — they prevent
re-running dead ends.

### 3. `experiments/{slug}/README.md` — Per-Arm Spec

One per experimental arm. Sections: Hypothesis, Differs from baseline by,
Expected outcome, Runs table.

## Project Verdict Vocabulary

State tags (`[ACTIVE]`/`[TBD]`/`[CONCLUDED]`) describe where a workstream is.
Use the following verdicts to describe the *decision*:

| Verdict | Meaning | Required artifact |
|---|---|---|
| **GO** | Hypothesis held; continue / scale / productionize. | Ledger entry with PASS + adversarial pass complete. |
| **PIVOT** | Core idea partially works; redirect to the part that does. | Ledger entry naming what worked vs what didn't + new direction. |
| **PARK** | Inconclusive, blocked on external input. | `PROJECT_STATUS.md` naming the exact unblock condition. |
| **KILL** | Hypothesis disproven or approach fundamentally unsuitable. | `DISCONTINUATION_NOTICE.md` (see below). **Non-negotiable.** |

## DISCONTINUATION_NOTICE.md — The Project Tombstone (KILL only)

When a workstream is KILLed, write a tombstone before archiving. This converts
sunk cost into permanent, reusable knowledge and stops the idea from being
blindly re-attempted.

```markdown
# Discontinuation Notice — {arm-slug}

**Date:** YYYY-MM-DD
**Workstream:** {arm-slug} ({one-line description})
**Status:** DISCONTINUED

## Summary
What was attempted and the top-line reason it was stopped.

## What We Learned
### Successful components ✅
- Concrete wins with numbers.
### Failed components ❌
- Concrete failures with numbers.

## Root Cause
The structural reason it can't work — the insight, not the symptom.

## Why We're Sharing This
What a future attempt should NOT do.

## Salvage
Which components/checkpoints/datasets are worth keeping and where they moved.
```

Record the KILL in the ledger (`02_EXPERIMENTS_AND_RESULTS.md`) and the tree
(`03_EXPERIMENT_TREE.md`).

## Sweep Management

For hyperparameter sweeps spanning multiple arms:

1. **One arm per config row.** Each distinct hyperparameter set is its own arm.
2. **Group conceptually in docs**, not in filesystem. The tree and ledger group
   related arms under one heading.
3. **Tracking table** in the ledger shows all arms side-by-side.
4. **Run sequentially on single GPU** — parallel training halves throughput and
   changes gradient dynamics.
5. **Chained sequential runs** (e.g., overnight GPU window): launch arm A, then
   run a polling script that auto-launches arm B when arm A's process exits.

### Epoch Budgeting for Frozen-Backbone Sweeps

Frozen backbone + small trainable head converges far faster than end-to-end:
- **Frozen backbone + deconv head:** 20–25 epochs. The head converges in 5–10
  epochs; more epochs waste compute on overfitting.
- **Full fine-tuning:** 50+ epochs.

Budget epochs by trainable parameter count, not habit.

## Process for Agents (AI and Human)

This section exists so that future agents (Hermes, Claude, Codex, etc.) and
new collaborators can follow the scientific process correctly without needing
to reverse-engineer conventions from existing files. **Follow these steps in
order when asked to run or record an experiment.**

### Before running ANY experiment

1. **Read `PROJECT_STATUS.md`** — know the current phase, blockers, and the
   single next action. Do not start new work without understanding the project
   state.
2. **Read `03_EXPERIMENT_TREE.md`** — check whether this experiment is already
   `[CONCLUDED]`. If it is, stop. Do not re-run a concluded experiment without
   explicit direction. Negative results (KILLed arms) are documented so they
   are NOT re-attempted blindly.
3. **Check the ledger (`02_EXPERIMENTS_AND_RESULTS.md`)** — look for prior
   related experiments. The ledger is the permanent record; trust it over your
   own assumptions about what should work.
4. **Read the existing code.** Before proposing a plan, inspect the relevant
   source files: what measurement machinery already exists, its inputs/outputs,
   and its known pitfalls. Also trace existing data-loading infrastructure for
   reusable extension points. A plan built without reading the code is a plan
   built on assumptions.
5. **Survey prior work.** Ground the question in what is already known — the
   established method, known confounds, prior effect sizes to gate against.
   Searching only the repo and your data misses the literature; an invented
   method where a standard exists, or a "novel" result that is a known artifact,
   is wasted effort.
6. **Survey available data BEFORE designing the pipeline.** Check all local
   storage for pre-extracted features, captions, and metadata. Match what exists
   against what the experiment actually needs.
7. **Verify the git branch is clean and on the correct base.**
   ```bash
   git status --short   # must be clean (or only untracked experiment files)
   git branch           # confirm the base is right, not a stale feature branch
   ```
   An experiment built on a dirty branch inherits unrelated changes — provenance
   says `git_commit: X` while the code is X + 10 dirty files from another arm.
   If the correct base lacks a dependency the arm needs, **say so explicitly**
   rather than silently branching from an unrelated `exp/*` branch.
8. **Create the arm's governance BEFORE touching any code:**
   - `provenance.yaml` with `mode`, hypothesis, falsified-if, and pre-registered gate
   - `config.yaml` with canonical parameters
   - `README.md` with hypothesis, differs-from, expected outcome
9. **Register the arm in the tree** (`03_EXPERIMENT_TREE.md`) as `[ACTIVE]`.
10. **Register the pre-registered gate in the ledger** (`02_EXPERIMENTS_AND_RESULTS.md`)
    BEFORE seeing results. The gate must be stated first — this is non-negotiable.

### When the experiment produces results

11. **Run the adversarial pass** on any candidate PASS. The full checklist (null
    computed? metric tested? metric stable? reproducible? extremes inspected?
    headline traced to an artifact? every flaw fixed or explicitly gated?) is NOT
    optional. Fill it in before writing the verdict. If any box is unchecked, the
    verdict is `PENDING`.
12. **Write the ledger entry** with: Goal, Pre-registered gate (copy from step 10),
    Empirical Evidence (numbers, not narrative), Adversarial Pass checklist,
    Verdict (GO/PIVOT/PARK/KILL).
13. **Update `provenance.yaml`** with `git_commit`, `data_snapshot`, and results
    in the `notes` field.
14. **Update the tree** (`03_EXPERIMENT_TREE.md`): move the arm from `[ACTIVE]`
    to `[CONCLUDED]` with the verdict.
15. **Update `PROJECT_STATUS.md`** with the new headline result and next action.

### When an experiment is KILLed

16. Write a `DISCONTINUATION_NOTICE.md` in the arm directory. This is the single
    most valuable artifact a dead experiment produces — it prevents re-attempt.
    It must state the **structural** reason the approach cannot work, not just
    that it failed. Marking something "dead" in the tree is not a tombstone.
17. Record the KILL in the ledger and tree, and cross-link the tombstone.
18. Never delete the code or artifacts. A KILLed arm with provenance is permanent
    knowledge; a deleted arm is a trap for the future.

### Critical rules agents MUST follow

- **Never re-run a `[CONCLUDED — FAIL]` or KILLed experiment** without explicit
  user direction and a new hypothesis. The ledger exists to prevent this.
- **Never skip the adversarial pass.** A measured PASS that was actually a
  measurement bug is the most expensive failure mode in private research. See
  the Phase 2b (z_a) overturn as a real example.
- **Never start a new experiment without reading `PROJECT_STATUS.md` first.**
  The project may be PARKed or between phases. Working on the wrong thing is
  worse than doing nothing.
- **Always verify identity test sets visually.** Contamination (name collisions,
  couple-shoot faces, seg-collapse) has nearly killed valid results before.
  See the Phase 1-R contamination near-miss.
- **Config keys that look right but aren't consumed by code produce silently
  invalid experiments.** grep-trace every config key through the codebase.
- **`git_commit` in provenance.yaml MUST be filled at run start.** A provenance
  record without a commit SHA is not reproducible.

## Common Pitfalls (project-specific)

- **Dead config keys.** Always `grep`-trace every config key to the code.
  A key that looks right but is never read produces silently invalid experiments.
- **CE loss ignoring class weights.** PyTorch `CrossEntropyLoss` needs an
  explicit `weight` tensor.
- **trace-J for concatenated vectors.** `tr(J) = tr(S_B)/tr(S_W)` is a weighted
  average — blind to complementarity. Use verification AUC for concatenated
  vectors.
- **Synthetic probes can be circular.** A probe that generates data with the same
  model it's testing will always pass. Use real-image gates.
- **Contamination in identity test sets.** Name collisions (e.g., `darina` vs
  `darina-l`), couple-shoot faces pulled by single-person detectors. Always
  visually verify identities in a gate set.
- **Metric definitions that drift between arms.** If the validator/scorer
  changed, re-run the baseline under the current definition.
- **Segmentation collapse on tight face crops.** Sapiens seg can collapse on
  ~10% of tight crops → empty masked vectors poison gates. Use fg≥30% filter.
