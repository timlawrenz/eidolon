# Eidolon Strategy Synthesis — Next Steps Recommendation

**Author:** Hermes Agent (eidolon session)  
**Date:** 2026-06-12  
**Sources:** `docs/reviews/strategy_review1.txt`, `strategy_review2.txt`, `strategy_review3.txt`  
**Inputs cross-checked against:** `docs/02_EXPERIMENTS_AND_RESULTS.md`, `docs/03_EXPERIMENT_TREE.md`, `docs/architecture.md`, gate result artifacts

---

## 1. What the Reviews Agree On (Consensus)

All three reviewers independently converged on the same core assessment:

| Consensus Point | Strength |
|---|---|
| The conditioning stack is **DINO patches (identity, AUC 0.797) + z_g (interpretable geometry)** | Unanimous |
| z_d (depth) and z_a (normals) are **definitively dead** — monocular volumetric models hallucinate generic, non-identity-specific geometry | Unanimous |
| DINO→slider bridge is dead in **both directions** (R² premise + identity transfer) | Unanimous |
| Phase 5 (DiT conditioning stack) is the next major build target | Unanimous |
| Block-diagonal ingestion + decoupled cross-attention (IP-Adapter style) with summed attention is the correct architecture | Unanimous |
| Pre-registered gates with proper null controls (random-projection null, permutation null) are essential methodology | Unanimous |
| Documentation needs updating to reflect the post-overturn state | Unanimous |

**No reviewer proposed reviving z_d or z_a, or continuing the DINO bridge.** Those paths are settled.

---

## 2. Where They Differ — And My Assessment

### 2.1 Sequencing: Jump to DiT vs. Measure First

| Reviewer | Position |
|---|---|
| Review 1 | Build Phase 5 DiT **now** with the proven 2-stream stack |
| Review 2 | Spike FLAME β first (~2-3 days), then DiT with what works |
| Review 3 | Close 5 measurement debts first (~1 week, CPU-only), then adapter-first spike |

**My assessment:** Review 3 is correct. Three of its five Tier 0 measurements could **fundamentally change** the Phase 5 design before any GPU cycles are spent:

- **ArcFace baseline (0.1):** If ArcFace AUC ≫ 0.797 on the hegre gate, the identity stream should be ArcFace (512-d), not DINO patches. This changes the ingestion architecture. The claim that "Eidolon's DINO patches are stronger than ArcFace" (Review 2) is unverified — ArcFace has never been measured on this gate.
- **Pose double-conditioning probe (0.2):** z_g was rebuilt at significant cost to be pose-invariant, yet the DINO stream of posed images carries pose intrinsically. If this contradiction isn't resolved, the architectural rationale for z_g's pose-invariance collapses. Review 1 and Review 2 completely missed this.
- **Morphology/transient slider split (0.3):** Phase 5's identity-consistency gates REQUIRE knowing which sliders are morphology (expected to change identity) vs. transient (must preserve identity). This split hasn't been computed, though the data exists.

### 2.2 FLAME β: Central Path vs. Optional Check

| Reviewer | Position |
|---|---|
| Review 2 | **Path B** — highest expected value. Replace dead monocular partitions with FLAME identity shape β. Concrete spike design with pre-registered gate. |
| Review 3 | "Optional cheap baseline" — buried at the end of Tier 2 |
| Review 1 | Only mentions 3DMMs as a resort if z_g is insufficient |

**My assessment:** Review 2's FLAME argument is the most **substantive architectural proposal** in any of the reviews. The logic is sound:

1. FLAME β is PCA-derived by construction (trained on 4D face scans) → satisfies Eidolon's orthogonality principle natively
2. It encodes identity-separated biological morphology, unlike Sapiens' generic geometry
3. There is direct literature precedent: FLAME blendshape parameters have been successfully used as diffusion conditioning via IP-Adapter-style decoupled cross-attention (ICCVW 2025)
4. The spike is cheap: off-the-shelf fitter (DECA/SMIRK/Pixel3DMM) + existing verification AUC gate = days, not weeks
5. If β passes the gate (AUC(β) > 0.6, ΔAUC > 0.01 when combined with z_g), Eidolon regains the **structured, model-based partition design** that was the original North Star — replacing the dead monocular path with a path backed by 30 years of 3DMM validation

**Caveats the reviews don't address:** FLAME fitters are known to struggle with editorial lighting and extreme expressions. The hegre corpus is exactly that. The "100+ reviewed identities" may yield noisy β estimates. But this is what the spike tests.

**Recommendation:** Promote FLAME β from "optional check" to **Priority 2** (after Tier 0 measurements). The spike gate is well-designed; if it fails, cost is days. If it passes, Eidolon has a structured geometric partition that satisfies its own principles.

### 2.3 From-Scratch DiT vs. Adapter-First

| Reviewer | Position |
|---|---|
| Review 1 | Custom DiT with decoupled cross-attention |
| Review 2 | "Build the DiT now with what works" |
| Review 3 | Adapter-first (IP-Adapter on frozen backbone like PixArt-α), escalate to custom DiT only if the frozen prior fights the firewall |

**My assessment:** Review 3's adapter-first argument is the economically honest one for a 4090. From-scratch DiT training on 70k FFHQ images (single-image-per-ID) will produce mediocre samples that muddy slider evaluation — and the identity-generalization problem (same person, new image) can't be solved on FFHQ regardless. An adapter on a frozen pretrained backbone:

- Costs ~22M trainable params (fits 4090 comfortably)
- Inherits a powerful image prior → clean samples for slider evaluation
- Validates the firewall/decoupled-attention thesis cheaply
- Only escalates to custom DiT if the frozen prior provably fights disentanglement

**Recommendation:** Adapter-first for Phase 5.1. Custom DiT is Phase 5.2, gated on adapter results.

### 2.4 Documentation Debt

Review 3's checklist (0.5) is the most specific: rewrite architecture.md §0/§4/§5/§7 to post-overturn state, add supersession markers on stale Phase 2b PASS, reunify orphaned Phase 2 evidence, preserve 2026-06-11 artifacts, define CLEAN/SUSPECT threshold, back-annotate literature synthesis.

**My assessment after reading architecture.md:** The document *has* been partially updated (header says "post-Phase-4 empirical record", z_d/z_a marked DEAD, n_parts changed to 1). But the supersession markers and orphaned evidence reunification are still needed. The literature synthesis has not been back-annotated. This is real debt.

---

## 3. Recommended Order of Operations

### Tier 0 — Measurement Debts (This Week, CPU-Only, No Training)

These are **pre-conditions** for designing Phase 5 correctly. Each could change the architecture.

| # | Action | Effort | Risk Retired | Source |
|---|---|---|---|---|
| **0.1** | **ArcFace/AdaFace gate baseline** on hegre face-crop corpus | ~½ day | Wrong identity signal chosen for Phase 5 | Review 3 §0.1 |
| **0.2** | **Pose-decodability probe on DINO stream** — linear-regress yaw/pitch from pooled flesh-masked vector. If pose is strongly decodable, document the decision to accept entanglement in the identity stream (as Arc2Face/IP-Adapter do) and retire "pose-orthogonal complement" as a system-wide requirement | ~½ day | Hidden double-conditioning contradiction | Review 3 §0.2 |
| **0.3** | **Morphology/transient slider tagging** — rank all 50 z_g components by within/between-identity Fisher J variance ratio (data exists in zd_gate_results.json). Tag each as morphology (high J) or transient (low J) | ~½ day | Untestable Phase 5 identity-consistency gates | Review 3 §0.3 |
| **0.4** | **Documentation catch-up** — supersession markers on stale Phase 2b PASS, reunify orphaned Phase 2 evidence, define CLEAN/SUSPECT threshold, back-annotate literature synthesis, preserve 2026-06-11 face-crop artifacts under docs/assets/ | ~1 day | Team acts on falsified claims | Review 3 §0.5 |
| **0.5** | **External pose-controlled corpus check** — FaceScape (free academic, 847 subjects × 20 expressions, real 3D GT) or CFP-style frontal-profile pairs. Test (a) z_g pose-invariance against real multi-view data and (b) DINO-vs-z_g ordering replicates off-hegre | Days (access wait) | Single-corpus universality assumption | Review 3 §0.4 |

### Tier 1 — Architectural Exploration (Before GPU Investment)

| # | Action | Effort | Gating Criteria | Source |
|---|---|---|---|---|
| **1.1** | **FLAME β spike** — DECA or SMIRK on hegre face-crop corpus → extract β identity shape params → run verification AUC gate: AUC(β_alone) and AUC([z_g \| β]). Pre-register: PASS if AUC(β) > 0.6 AND ΔAUC > 0.01 | 2–3 days | If PASS: freeze β encoder, integrate as third structured partition. If FAIL: β is not identity-bearing on this corpus; z_g remains sole structured partition | Review 2 Path B |
| **1.2** | **Pre-register Phase 5 gates** in ledger with Phase 4-grade methodology: slider obedience (cycle consistency), transient/morphology dissociation, conflict test (identity from A, z_g from B → geometry tracks B, appearance tracks A), λ=0 arm, shuffled-z_g null | ~½ day | Prevents post-hoc gate design | Review 3 §1.4 |
| **1.3** | **Solve the identity-data problem** — decide reconstruction-training (accept v1 limitation) vs. acquire multi-image-per-ID corpus. Do NOT train on hegre (gate contamination) | Decision + data prep | Determines scope of Phase 5 claims | Review 3 §1.3 |

### Tier 2 — Phase 5 Build (Adapter-First)

| # | Action | Effort | Source |
|---|---|---|---|
| **2.1** | **Adapter-first conditioning spike** — IP-Adapter-style decoupled cross-attention on frozen PixArt-α or SD-class backbone. 2 or 3 streams depending on FLAME β outcome. λ sliders. Conflict test as headline validation | 1–2 weeks | Review 3 §1.1 |
| **2.2** | **Redesign §7 ingestion** for asymmetric streams: ~50 z_g scalars (expanded tokens) + ~1,261 DINO patch tokens (full set or perceiver resampler) + possibly one ArcFace token or FLAME β stream. Drop dead mlp_d. | Integrated into 2.1 | Review 3 §1.2 |
| **2.3** | **Escalate to custom DiT** ONLY if 2.1 proves the frozen prior fights the firewall. If escalated: REPA-style alignment with cached DINOv3 targets. | Weeks | Review 3 §1.1 |

### Tier 3 — Positioning & Wildcards

| # | Action | Source |
|---|---|---|
| **3.1** | **Re-scope narrative** — the "unique intersection" claim (three PCA modalities + firewall + DiT) no longer describes the project. Defensible claims: (1) pose-invariant orthogonal keypoint slider space, gate-validated; (2) firewalled multi-stream conditioning with per-stream λ; (3) quantified slider-obedience/conflict metrics; (4) disciplined negative-results record on monocular volumetrics | Review 3 §Tier 2 |
| **3.2** | **Write up negative results** — the z_d/z_a kill chain is a publishable methods/negative-results piece. Soften "monocular models hallucinate generic geometry" to "adds no verification signal under PCA-k50 on our corpus" unless a raw-pixel/high-k control is run (only 66–75% variance was retained) | Review 3 §Tier 2 |
| **3.3** | **RAE-style diffusion spike** (wildcard, exploratory) — DINOv3 latent-space DiT. Stratum already stores DINOv3 patches for the corpus; decoder training is 4090-feasible at 256px. Branch-sized bet, not mainline | Review 3 §1.5 |

---

## 4. What Specifically Each Review Got Right, Wrong, or Incomplete

### Review 1 ("Specialist 1")

**Right:** Correctly identifies the pivoted stack. The disentanglement gate design (§3) is well-specified. The warning about masked-Linear leakage in block-diagonal ingestion aligns with architecture.md §7.1.

**Wrong/Incomplete:** Jumps straight to DiT without acknowledging that key measurements are missing. Recommends "Holtz et al. unsupervised disentanglement metric" as a gate — this metric was mentioned in the literature synthesis, never implemented, and Review 3 correctly notes it should either be used or struck. The methodology overhaul recommendation ("strip manifesto language") is valid in principle but the docs are already more honest than Review 1 seems to assume.

**Unique contribution:** The detailed test design for the disentanglement gate (DINO constant + z_g traversal → morphological change without identity shift, and vice versa).

### Review 2 ("Specialist 2")

**Right:** The FLAME β argument is the most important architectural proposal in any review. The Path A/B/C/D framework provides clear decision criteria. The concrete spike design (DECA/SMIRK, pre-registered gate, 2-3 day estimate) is actionable. Correctly identifies that FLAME β is PCA-derived and thus satisfies Eidolon's orthogonality principle natively.

**Wrong/Incomplete:** States "Eidolon's DINO patches (AUC 0.797) are a stronger identity carrier than ArcFace embeddings" — this is unmeasured conjecture. ArcFace has never been run on the hegre gate. If ArcFace AUC is > 0.9 (as Review 3 predicts), the identity stream should be ArcFace, not DINO patches. The "immediate" recommendation to skip Tier 0 measurements and go straight to FLAME β + DiT is premature. Multi-view audit (Path C) is interesting but should be a background task, not a blocker.

**Unique contribution:** The FLAME β → E = [z_β | z_ψ | z_θ] proposal. The corrected North Star vision (Eidolon v2 with FLAME β + z_g). The ICCVW 2025 blendshape-conditioned diffusion reference that proves 3DMM parameters work as diffusion conditioning.

### Review 3 ("Specialist 3")

**Right:** The Tier 0 → Tier 1 → Tier 2 framework is the right ordering. Every measurement debt identified (0.1–0.5) is real and important. The pose double-conditioning contradiction (0.2) is the single sharpest critique — it catches something both other reviews missed. The adapter-first recommendation is economically honest for the 4090. The pre-registered Phase 5 gate proposals are specific and inherit Phase 4's methodological rigor. The RAE wildcard (1.5) is correctly framed as exploratory, not mainline.

**Wrong/Incomplete:** Undersells the FLAME β opportunity — buries it as an "optional cheap baseline" when it's actually the most promising path to recovering structured model-based partitions. The prediction that "ArcFace will very likely land ≥0.9" is plausible but stated with more confidence than the evidence supports (editorial face crops with heavy makeup, lighting, and expressions could suppress FR embedding performance). The "RAE-style diffusion" recommendation is interesting but adds scope to an already full plate.

**Unique contribution:** The ArcFace baseline gap identification (most important missing number). The pose double-conditioning probe. The morphology/transient slider decomposition. The specific, actionable documentation debt checklist. The adapter-first economic argument.

---

## 5. The Bottom Line

**This week:** Run Tier 0 measurements (0.1–0.4). These are all CPU-cheap and none can be skipped — the ArcFace number alone could redirect the identity stream. If ArcFace AUC > 0.85, the DINO-patches-as-identity story needs to be revisited.

**If Tier 0 confirms the current stack:** Run the FLAME β spike (1.1). It's a 2–3 day bet on recovering the structured-partition North Star. If β passes, Eidolon has a real third partition grounded in 30 years of 3DMM research. If it fails, z_g remains the sole structured partition and Phase 5 proceeds with 2 streams.

**After measurements + FLAME decision:** Build the adapter-first Phase 5 spike (2.1) with pre-registered gates (1.2). Validate the firewall, slider obedience, and conflict test on a frozen backbone before committing to from-scratch DiT training.

This sequencing de-risks every major decision before GPU compute is spent. It respects the methodology the project has already invested in — pre-registered gates, random-projection nulls, honest negative results — while acting on the most promising new evidence (FLAME β as a structured partition).
