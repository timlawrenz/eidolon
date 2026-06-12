# Documentation Sync — Post-Phase-4 Ground Truth

> **For Hermes:** Execute directly (not via subagent) — this is a text-editing task, not code.
**Goal:** Bring all four docs into consistency with the post-Phase-4 empirical record, where E's structured partition is z_g alone and identity comes from DINO patch tokens.
**Architecture:** Targeted edits to 4 markdown files + archive annotation of stale JSON artifacts. No new files except possibly a consolidated summary section.
**Three-agent review input:** Synthesized from reviews attached to the user's message — all three converge on `architecture.md` staleness, ledger structural decay, and missing consolidated summary.

---

## Current State (ground truth to propagate)

- **z_g** (geometry encoder): shipped, frozen, pose-invariant, AUC ≈ 0.67–0.69 (face-crop). The sole surviving structured partition.
- **z_d** (depth): dead. Confirmed at 24× resolution with domain shift eliminated. ΔAUC −0.023 to −0.034.
- **z_a** (normals): dead. Initial PASS overturned when z_g baseline corrected from 0.540 → 0.688. ΔAUC −0.039.
- **DINO bridge**: dead in both directions. R² fails (0.385 for z_a). Transfer loses identity below random projections.
- **Phase 4 (masked DINO patches)**: PASS. Flesh-masked mean-pooled patches AUC 0.797, cross-shoot verified, Δ+0.027 over cls.
- **Settled conditioning stack**: flesh-masked DINOv3 patch tokens (identity) + z_g (interpretable geometry control).
- **Phase 5 (DiT fusion)**: TBD — next build target.

---

## Task 1: Rewrite `architecture.md` to reflect post-Phase-4 truth

**Files:** `docs/architecture.md` (421 lines)

**Objective:** The definitive design document must describe the architecture as *empirically validated*, not as originally designed.

### Step 1.1: Update §0 North Star

- Change `E = [z_g | z_d | z_a]` to reflect current state.
- Mark z_d and z_a as dead partitions with clear annotations.
- Update the "albedo / surface" language to "normals / surface (DEAD)".
- Show the current 2-path conditioning stack.

**New text for §0:**
```
E = [z_g]   ∈ ℝ^50  (structured, interpretable geometry control)
      │
      └────────────── geometry / shape  (z_g from facial keypoints, pose-invariant)

Identity conditioning: flesh-masked DINOv3 patch tokens (external to E, AUC 0.797)
Former partitions: z_d (depth, DEAD) and z_a (normals, DEAD) — both conclusively
  failed verification-AUC gate; monocular volumetrics hallucinate generic human geometry.
```

### Step 1.2: Update §4 Volumetric Encoders

- Change z_a tag from `(ACTIVE)` to `(DEAD)`.
- Add one-sentence verdict with pointer to ledger.
- Keep the 2.5D rotation trap warning (it's still valid architectural knowledge).
- Remove or annotate the "Macro architectural finding" paragraph (lines 227–233) — every claim in it was overturned.

**Action:** Replace lines 187–233 with condensed, verdict-aware text:
- Mark both z_d and z_a as dead.
- Preserve the 2.5D rotation trap as a permanent architectural warning.
- Recategorize the "Macro architectural finding" as `[OVERTURNED 2026-06-11]` with a brief explanation and pointer to ledger Phase 2b face-crop overturn.

### Step 1.3: Update §5 DINOv3 Bridge

- Remove "The R² of this regression is the single most important number in the project."
- Document that both fast-path directions are dead.
- Record the random-projection null lesson.
- Note that raw DINO carries identity (AUC 0.766/0.797) but cannot reconstruct sliders.

### Step 1.4: Update §6 Scoring

- Simplify: only z_g scoring survives in E's structured partition.
- Add note that identity comes from DINO patch extraction (external to E).

### Step 1.5: Update §7 DiT Fusion

- Change `n_parts=3` to `n_parts=1` (z_g only) in `BlockDiagonalIngestion`.
- Update prose: identity from DINO cross-attention, control from z_g.
- Document the 2-stream design, not 3-stream.
- Keep the block-diagonal + AdamW weight-decay leak explanation (it's still valuable).

### Step 1.6: Update §8 Build Order

- Mark Phases 1–4 as concluded.
- Phase 5 (DiT conditioning stack) as the only open item.
- Remove obsolete "Phase 1 is the first build target" framing.

### Step 1.7: Minor fixes

- Fix `C1 raw variance ≈ 1000` — this is impossible for [-1,1]-normalized coordinates. Remove or replace with an actual measured value.
- Change "albedo" → "normals/surface" throughout. Albedo is reflectance; these encoders use surface normals.

---

## Task 2: Clean up `02_EXPERIMENTS_AND_RESULTS.md`

**Files:** `docs/02_EXPERIMENTS_AND_RESULTS.md` (696 lines)

**Objective:** Fix structural defects without altering the empirical record.

### Step 2.1: Remove duplicate Phase 3 stub (lines 367–395)

The `[ACTIVE]` Phase 3 section at line 367 is a pre-registration stub that was superseded by the `[CONCLUDED]` Phase 3 section at line 398. The stub has a `[PENDING]` verdict with no results. The concluded section has the actual stratified verdict.

**Action:** Either:
- Delete the stub entirely (lines 367–395), OR
- Merge its pre-registered gate criteria into the concluded section's header and delete the stub.

Prefer deleting the stub and moving any unique content (the pre-registered gate descriptions) into the concluded section if they're needed there.

### Step 2.2: Fix Phase 2b verdict header (line 323)

**Current:** `[CONCLUDED — z_a PASSES] (2026-06-10).`

**Change to:** `[CONCLUDED — FAIL (initial PASS overturned 2026-06-11)]`

Add a one-line pointer: "The PASS below was overturned by the face-crop re-run. See [2026-06-11 UPDATE] Phase 2b (Normals) Face-Crop OVERTURN at line 571."

### Step 2.3: Audit status tags across the document

Search for all `[ACTIVE]` / `[PENDING]` / `[CONCLUDED]` tags and verify each matches reality:

| Section | Current tag | True status | Action |
|---------|------------|-------------|--------|
| Phase 1-R (line 72) | `[ACTIVE]` | CONCLUDED | Fix |
| Phase 2 (line ~230) | `[ACTIVE]` | CONCLUDED — FAIL | Fix |
| Phase 2b (line 323) | `[CONCLUDED — z_a PASSES]` | FAIL (overturned) | Fix |
| Phase 3 stub (line 367) | `[ACTIVE]` | Remove or merge | Remove |
| Phase 3 (line 398) | `[CONCLUDED]` | CONCLUDED | OK |
| Phase 4 (line 595) | `[ACTIVE]` | CONCLUDED — PASS | Fix |
| Metric fix (line 695) | `[ACTIVE]` | CONCLUDED (implemented) | Fix |

### Step 2.4: Reorganize z_d evidence (Phase 2 section)

The z_d story spans ~200 lines with the gate run, metric bug, cross-examination, face-crop re-run, z_g correction, and z_a overturn all interleaved. Options:
- **Minimal:** Add section anchors/ToC so a reader can navigate the 5 subsections.
- **Moderate:** Move the forensic post-mortems (trace-J bug analysis, cross-examination table, seg-collapse defect) into collapsed `<details>` blocks or appendices to reduce scroll depth. Keep the verdict, key numbers, and pointers in the main flow.
- **Heavy:** Extract post-mortems into separate appendix notes in `docs/appendix/`. Not recommended for this pass — increases document count without solving consistency.

**Recommend minimal** for this pass: add clear section break markers and a 2-line summary at the top of the z_d section pointing to each sub-section.

### Step 2.5: Add consolidated "Current State" section at top of ledger

After the intro (line 3), insert a "## Current State (as of 2026-06-11)" section answering:

- **What is E today?** z_g (50-d pose-invariant geometry), external DINO patch identity.
- **What survived?** z_g (Phase 1-R), masked DINO patches (Phase 4).
- **What died?** z_d (Phase 2), z_a (Phase 2b), DINO bridge (Phase 3).
- **Next build target?** Phase 5: DiT fusion stack.
- **Key methodological lessons:** Gate-metric cross-examination, random-projection null requirement, measurement-resolution baseline trap, seg-collapse detection.

This should be ≤15 lines — a quick orientation for any reader.

---

## Task 3: Update `01_LITERATURE_SYNTHESIS.md`

**Files:** `docs/01_LITERATURE_SYNTHESIS.md` (178 lines)

### Step 3.1: Add monocular hallucination literature context

The synthesis currently has no literature anchor for the project's central negative finding: monocular volumetric models hallucinate generic human geometry. Add a subsection or paragraph connecting to the monocular depth estimation limits literature.

**Action:** Add to the Critical Assessment (§7) or as a new subsection: note that the literature on monocular depth estimation for biometric identity is thin but relevant, and that the project's empirical finding (Sapiens normals/depth don't carry identity-specific signal) is a contribution to this gap.

### Step 3.2: Update outdated "3-modality" framing

- §6: "Whether 3-modality ingestion (z_g, z_d, z_a) is sufficient" → update to note that experiments show it was one modality too many.
- §3 (Competitor Landscape): Eidolon's differentiator is no longer "3 modalities" — it's "PCA-guaranteed disentanglement + DINO semantic identity + block-diagonal firewall."

### Step 3.3: Ground Phase 4 masked-patch approach

Add a reference or note connecting the flesh-masked DINOv3 patch pooling to relevant literature on masked attention / region-pooled ViT features for face recognition.

### Step 3.4: Tone down validation claims

- "The stratum-hq → Sapiens pipeline for z_d is validated" → update: the pipeline produced valid topology, but the resulting features carry no identity signal.
- "VAEs Pursue PCA Directions" claim: add the caveat that Rolinek shows tendency under specific conditions, not universal proof that PCA ≥ learned latents.

---

## Task 4: Update `03_EXPERIMENT_TREE.md`

**Files:** `docs/03_EXPERIMENT_TREE.md` (69 lines)

**Objective:** Minor updates for completeness.

### Step 4.1: Add Phase 4 to "Active & Planned" transition

Phase 4 appears only under "Concluded." Add a note or show its lifecycle (planned → active → concluded).

### Step 4.2: Elevate conditioning stack to standalone top-level statement

The settled stack is currently buried in the Phase 5 entry's parenthetical. Add a "## Current Architecture" section or prominent top-line statement:

```
## Settled Conditioning Stack

Identity: flesh-masked DINOv3 patch tokens (Phase 4, AUC 0.797, cross-shoot verified)
Control:  z_g (Phase 1-R, 50-d pose-invariant geometry encoder)
Dead:     z_d (depth), z_a (normals), DINO→slider bridge
Next:     Phase 5 — DiT fusion stack with 2-stream decoupled cross-attention
```

### Step 4.3: Add cross-references

Link each concluded phase back to the relevant section in `02_EXPERIMENTS_AND_RESULTS.md` and `architecture.md`.

---

## Task 5: Annotate stale artifacts

**Files:** JSON artifacts under `docs/assets/exp/geometry-pca/`

**Objective:** Artifacts that contradict final verdicts need annotation so future readers aren't misled.

### Step 5.1: Audit artifact verdicts vs. final verdicts

| Artifact | Says | Final truth | Action |
|----------|------|-------------|--------|
| `za_gate_results.json` | `"overall_verdict": "PASS"`, `"best_variant": "rot"` | FAIL, xy selected not rot | Annotate |
| `phase3b_transfer_results.json` | `"verdict": "PASS"`, `"fraction_identity_retained": 1.71` | UNINFORMATIVE (bridge ≤ random) | Annotate |
| `phase3_bridge_results.json` | (R² values) | OK — these are the correct measurements | No action |
| `zd_verification_auc.json` | (AUC values) | OK — these support the FAIL | No action |

### Step 5.2: Annotation method

**Option A (preferred):** Add a `_README.md` in `docs/assets/exp/geometry-pca/` listing each artifact's status against final verdicts, with dates and pointers to the ledger.

**Option B:** Modify each JSON to add an `_annotation` key. Avoid — modifying evidence JSONs is bad practice.

**Option C:** Rename stale files with a `STALE_` prefix. Avoid — breaks script references.

**Choose Option A.** Create `docs/assets/exp/geometry-pca/ARTIFACT_STATUS.md` with a table mapping each artifact to its current evidentiary status.

### Step 5.3: Note missing Phase 4 artifacts

Phase 4 has zero persisted artifacts under `docs/assets/`. The ledger tables are the sole record. If `data/phase4_patch_pooling.json` exists elsewhere, copy it into the assets tree. If not, document that Phase 4 evidence exists only in the ledger.

---

## Task 6: Naming and terminology fixes

**Files:** `architecture.md`, `02_EXPERIMENTS_AND_RESULTS.md`

**Objective:** Fix "albedo" → "normals/surface" and other terminology issues.

### Step 6.1: "Albedo" → "normals" or "surface"

Search all four docs for "albedo" and replace with "surface normals" or "surface" as appropriate. Albedo is reflectance (intrinsic color). These encoders use surface normals (geometric orientation). This is not pedantry — it matters for understanding what the partition actually captured.

### Step 6.2: "North Star — face, albedo, and body"

Update to reflect that scope narrowed from "person" to "face" — nothing in the project touches body keypoints. Change to "face shape and identity."

### Step 6.3: Fix duplicate "3." numbering

In Phase 3 findings list (ledger, around line 448), there are two items numbered "3." Fix the second to "4."

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `architecture.md` rewrite is too aggressive — loses valuable design rationale | Preserve the block-diagonal ingestion explanation, 2.5D rotation trap, and AdamW weight-decay analysis unchanged. Only update stale factual claims. |
| Ledger cleanup accidentally alters empirical record | All changes are structural (tags, headers, section markers) or additive (summary section). No evidence tables or numbers modified. |
| Literature synthesis update requires new research | Keep scope bounded: flag the gap, add 1–2 sentence context, don't attempt a full new literature review. |
| Stale artifact annotation creates confusion | Use a single README, not per-file modifications. Clear table format with dates and ledger pointers. |

---

## Validation

After all edits:
- [ ] Read `architecture.md` start to finish — does a new reader get the correct current-state picture?
- [ ] `grep -n "\[ACTIVE\]" docs/02_EXPERIMENTS_AND_RESULTS.md` — only sections genuinely in progress should show.
- [ ] `grep -rn "albedo" docs/` — should return zero results (except possibly in literature references where it's correct).
- [ ] Cross-check `architecture.md` §0 North Star against `03_EXPERIMENT_TREE.md` settled stack — must match.
- [ ] Phase 2b verdict header matches body conclusion.
- [ ] Only one Phase 3 section remains.

---

## Execution order

1. **Task 2 first** (ledger cleanup) — establishes the canonical record that architecture.md will reference.
2. **Task 1** (architecture.md rewrite) — the biggest effort, references the cleaned ledger.
3. **Task 4** (experiment tree update) — quick, builds on both.
4. **Task 3** (literature synthesis) — independent of 1/2/4, can run in parallel.
5. **Task 5** (artifact annotation) — last, after text docs are settled.
6. **Task 6** (naming fixes) — sprinkle across all tasks, or do as a final grep-and-replace pass.
