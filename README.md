# Project Eidolon

Welcome to the Eidolon research repository. This repository implements strict separation between core infrastructure and experimental paths.

> **🤖 AI agents: stop here and read [AGENTS.md](AGENTS.md) first.**
> It contains mandatory rules, reading order, and the scientific process you
> must follow before running any experiment or making any changes. This README
> is the human-facing overview.

To understand the current state, architecture, and history of the project, please start with the canonical documentation in the `docs/` folder.

## Quick Orientation

Start with **[Project Status](../PROJECT_STATUS.md)** — the living pointer to
what's happening right now and the single next action.

## Core Documentation (The Front Door)

1. **[Git Workflow & Experiment Isolation](docs/00_GIT_WORKFLOW.md)**
   Defines the branch-to-experiment mapping rules. Read this before checking out branches or starting new runs.
2. **[Vision and Architecture](docs/01_VISION_AND_ARCHITECTURE.md)**
   The single canonical source of truth for the project's conceptual foundation, architecture decisions, and theoretical bottlenecks.
3. **[Experiments and Results Ledger](docs/02_EXPERIMENTS_AND_RESULTS.md)**
   A historical ledger of what was tried. Includes explicit documentation of negative results (e.g., failed methods, mode collapses) backed by empirical proof.
4. **[Experiment Tree (Roadmap)](docs/03_EXPERIMENT_TREE.md)**
   A living, high-level strategic roadmap acting as a Kanban board with status tags (`[ACTIVE]`, `[NEXT]`, `[TBD]`, `[CONCLUDED]`). Links directly to the relevant `exp/*` git branches.
5. **[Experiment Directory Structure](docs/experiment-structure.md)**
   The governance contract — rules for running experiments, pre-registered gates, adversarial pass checklist, verdict vocabulary, and directory layout.

## Infrastructure vs. Experiments

*   The `main` branch contains only finalized documentation, infrastructure, and validated tools.
*   All active research and experimental code runs on isolated `exp/*` branches. Check the [Experiment Tree](docs/03_EXPERIMENT_TREE.md) for the active branch mapping.