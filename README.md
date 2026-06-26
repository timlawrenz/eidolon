# Project Eidolon

Welcome to the Eidolon research repository. This repository implements strict separation between core infrastructure and experimental paths.

To understand the current state, architecture, and history of the project, please start with the canonical documentation in the `docs/` folder.

## Core Documentation (The Front Door)

1. **[Git Workflow & Experiment Isolation](docs/00_GIT_WORKFLOW.md)**
   Defines the branch-to-experiment mapping rules. Read this before checking out branches or starting new runs.
2. **[Vision and Architecture](docs/01_VISION_AND_ARCHITECTURE.md)**
   The single canonical source of truth for the project's conceptual foundation, architecture decisions, and theoretical bottlenecks.
3. **[Experiments and Results Ledger](docs/02_EXPERIMENTS_AND_RESULTS.md)**
   A historical ledger of what was tried. Includes explicit documentation of negative results (e.g., failed methods, mode collapses) backed by empirical proof.
4. **[Experiment Tree (Roadmap)](docs/03_EXPERIMENT_TREE.md)**
   A living, high-level strategic roadmap acting as a Kanban board with status tags (`[ACTIVE]`, `[NEXT]`, `[TBD]`, `[CONCLUDED]`). Links directly to the relevant `exp/*` git branches.

## Infrastructure vs. Experiments

*   The `main` branch contains only finalized documentation, infrastructure, and validated tools.
*   All active research and experimental code runs on isolated `exp/*` branches. Check the [Experiment Tree](docs/03_EXPERIMENT_TREE.md) for the active branch mapping.