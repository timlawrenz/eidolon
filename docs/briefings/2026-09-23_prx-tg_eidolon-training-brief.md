# Briefing — Eidolon → prx-tg: which arms train on Eidolon's data, on what data, against which gates

| | |
|---|---|
| **From** | Eidolon agent (`~/source/activity/eidolon`) |
| **To** | prx-tg agent (`~/source/activity/prx-tg`) |
| **Carried by** | Tim (will walk through implementation details in person) |
| **Date** | 2026-09-23 (identity status revised 2026-09-24 — §2) |
| **Status** | **Discussion input only.** Nothing here is a registered prx-tg gate. Any gate below must be pre-registered in prx-tg's own ledger *before* the run it judges. |

---

## 0. Purpose

Three things:

1. **Separate the tracks.** `latent-first-pretrain` (P1) and `pixel-posttrain` (PP) are **not** Eidolon arms, and the taint diagnosis on them is a result about a *different* experiment.
2. **Name the arms that do train on Eidolon's desired data**, and what is actually in their training mix.
3. **Report the state of the conditioning streams** in that mix — identity is settled, `z_g` is still open — then propose data, gates and expectations for the next Eidolon-relevant arm.

---

## 1. The distinction that must hold: P1/PP are a training-mechanics spike, not Eidolon arms

Verified directly from the configs (`experiment-configs/*/config.yaml`):

| | `latent-first-pretrain` (P1) | `pixel-posttrain` (PP) |
|---|---|---|
| `data.source` | `stratum` | `stratum` |
| `data.stratum_dir` | `$STRATUM_DIR` (FFHQ) | `$STRATUM_DIR` (FFHQ) |
| `stratum_max_samples` | 70,000 | 70,000 |
| `model.in_channels` | **16** (FLUX-AE latent) | **3** (RGB pixel) |
| `latent_mode` | `true` (reads `flux_latent.npy`) | absent |
| `pixel_range` | — | `"-11"` (zero-mean → [-1,1]) |
| sampling input size | 128 (= 1024 / 8× AE) | 1024 |
| body | NanoDiT 768/18/12 | NanoDiT 768/18/12 |

**P1 and PP differ essentially only in *where* the diffusion happens** — FLUX VAE latent space vs direct pixel space. Same data (FFHQ stratum), same transformer body. That is precisely a **training-mechanics spike: "does latent-first pretraining pay off vs direct pixel training?"**

They do **not** use the Eidolon adapter, the hegre corpus, the AuraFace-LDA identity stream, or Eidolon's `z_g` geometry stream.

> **So the phrase "P1/PP were trained on AuraFace + DINO + pose" is wrong, and must not be used.** P1/PP are a VAE-vs-pixel training-mode experiment. Their taint finding (text perceptually dead, CLS used as a per-image lookup key) is a genuinely important prx-tg result — but it is a result about *that* track, and Eidolon should not inherit its diagnosis or its remedies by association.

### 1.1 The arms that *do* train on Eidolon's data

All on branch `exp/eidolon-conditioning`, all with `adapter.name: "eidolon"`, `dino_patches_enabled: false`:

| Arm | Directory | Status |
|---|---|---|
| Eidolon Conditioning baseline | `experiments/eidolon-conditioning/` | `[CONCLUDED — GO]` |
| Hegre Geometry | `experiments/hegre-geometry/` | `[CONCLUDED — GO]` |
| Arm N — `zg-token-basis` | `experiments/zg-token-basis/` | `[CONCLUDED — KILL]` (tombstone) |
| **Arm O — `zg-token-basis-cfg-guard`** | `experiments/zg-token-basis-cfg-guard/` | **`[CONCLUDED — PASS]`** ✅ |

Current best Eidolon-relevant result: **Arm O** — CFG-guarded per-dim basis. Monotonic `dim0→yaw` at steps 3000 **and** 5000, zero collapse through 5,000 steps, loss 0.0099, recon LPIPS 0.819. Release artifact `release/prx_tg_armo_cfgguard_step5000_bf16.safetensors`. This is the first arm ever to hold controlled yaw without collapse, and it validated the Arm N collapse diagnosis (basis injected on CFG-dropped steps = pure noise accumulation).

---

## 2. Identity stream status — one basis, one convention

All identity data consumed by the Eidolon adapter sits on the pooled refit basis
(`auraface_lda.npz`, refit 2026-07-23), L2-normalized to norm 1.0, and is stamped
with basis fingerprint **`120e1c5a1dc4f423`**:

| Directory | `auraface_lda` convention | stamp |
|---|---|---|
| `/mnt/nas-ai-models/training-data/eidolon/hegre_corpus` | refit basis + L2-normalize (norm 1.0) | ✅ |
| `/mnt/nas-ai-models/training-data/ffhq/stratum` | refit basis + L2-normalize (norm 1.0) | ✅ |
| `/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/lda` | refit basis, raw coords (norm ~153) | ✅ |

Verify any of them with:

```
hegre-dataset basis-fingerprint verify --dataset <dir>
```

`hegre-faces/v1/lda` is the per-image retrieval tree, not DiT input — raw coords
are its correct convention. **The two conventions must not be mixed in one slot:**
a DiT identity input takes the normalized form.

### What the model receives in the `identity_dim: 64` slot

| | FFHQ (weight 2.3) | hegre corpus (weight 1.0) |
|---|---|---|
| basis | refit | refit |
| vector norm | **exactly 1.000** | **exactly 1.000** |
| semantic level | **per-image** (each image its own identity) | **per-persona** (one vector per persona) |

One mismatch remains, and it is **semantic, not encoding**. FFHQ is predominantly
one image per identity, so its identity vectors teach "identity vector = per-image
key" (the CLS-memorization hazard class). FFHQ is encoding-consistent and usable
as a geometry/texture regularizer with the identity stream masked — it is **not** a
valid identity target.

### Standing consequence for the prior arms

The five `exp/eidolon-conditioning` arms trained before 2026-09-24, when FFHQ's
vectors were still on the pre-refit basis (norm 0.35) while hegre's were on the
refit basis (norm 1.0) — a 2.84× magnitude gap in the same 64-d slot, verified
bit-exact rather than inferred.

**Their identity conditioning is unsound. Draw no identity conclusion from any of
them.** Arm O's PASS is safe as a *geometry* result (`z_g` is a separate stream,
unaffected) — but its identity stream is untested, and Arm O's own validation
disabled the identity probe (see §4.1).

---

## 3. What Eidolon asks to be trained on

### 3.1 The conditioning contract (fixed — this is the shipped inference interface)

| Stream | Dim | Production path | Semantics | Injection |
|---|---|---|---|---|
| **identity** | 64 | AuraFace 512-d → nuisance-clean (remove PC1 domain axis + yaw direction) → LDA basis refit 2026-07-20 | **per-persona**, one vector shared by all images of that persona | adaLN |
| **geometry** | 50 | DWPose → `z_g`, identity-blind | **per-image** | cross-attention, per-dim basis (Arm O fix) |
| **shape** | TBD | Sapiens2 dense keypoints | per-image | Phase 5 open question |
| **text / T5** | — | **excluded** | — | — |
| **DINO patches / CLS** | — | **excluded** | — | dead partition |

Target: pixel 1024², `x_prediction`.

**Why text and DINO are excluded:** Eidolon's inference contract carries identity + geometry only. Training on channels that will not exist at inference is exactly how a model learns to route through the easiest available channel instead of the one you intend. This is the same logic that makes the P1/PP CLS-memorization failure mode reachable.

### 3.2 Data recommendation

**Primary: `hegre_corpus` only** — `/mnt/nas-ai-models/training-data/eidolon/hegre_corpus`
- 31,711 samples / 321 personas, single basis, manifest-verified
- `_manifest.json` basis fingerprint `e2f66241288e1f50`, on-disk == manifest exactly, 0 errors
- ~100 images per persona → a genuine persona-level identity target
- clean persona holdout available (321 personas) for a never-trained-identity gate

**FFHQ should be excluded from identity-conditioned training**, for one structural reason (§2 covers its encoding status):

**It cannot supply a persona-level identity target at all.** Measured on 3,000 FFHQ raw AuraFace vectors, nearest-neighbour cosine distribution: mean 0.373, p95 0.542. Vectors with any close partner:

| threshold | pairs | vectors involved |
|---|---|---|
| cos > 0.5 | 319 | 8.5% |
| cos > 0.6 | 133 | **3.2%** |
| cos > 0.7 | 37 | **1.3%** |

FFHQ is predominantly **one image per identity** — only 3.2% of sampled vectors have any partner above cosine 0.6, versus hegre's ~100 images per persona. It has no persona grouping to average over. So any FFHQ sample carrying an identity vector teaches the model that *the identity vector is a per-image key* — structurally identical to the CLS-as-lookup-key mechanism that tainted P1/PP. In the current mix that is **70% of samples by weight**.

**Options for discussion:**

| Option | Description | Assessment |
|---|---|---|
| **A (recommended)** | hegre_corpus only | Cleanest. Persona-level identity, no per-image key, clean persona holdout. Loses FFHQ's diversity/regularization. |
| B | FFHQ included as a **geometry/texture regularizer with the identity input masked** for FFHQ samples | Keeps diversity, removes the per-image-key hazard. Encoding is already consistent (§2). Needs an implementation decision on how to mask identity per-sample. |
| C | FFHQ included **with** identity | Not recommended as primary — reintroduces the per-image-key hazard. Only admissible if the G4 memorization gate passes. |

**A useful fact for the decision:** the persona-average identity index is sound and *better* than the per-image ceiling. Querying a held-out image's per-image LDA against an index of the 321/325 persona averages:

| metric | value |
|---|---|
| persona-average index R@1 | **0.8879** |
| R@5 | 0.9502 |
| R@10 | 0.9657 |
| chance R@1 | 0.0031 |

(vs. the recorded per-image GT-LDA ceiling R@1 = 0.8538). So the persona-level target the corpus ships is *more* discriminative than the per-image vectors the earlier ceiling was measured on.

**Caveat to carry into training:** that discriminative signal is small in absolute terms. Persona averages are 99.53% collinear pairwise (min 0.9872, max 0.9997); within-persona per-image cosine 0.9977 vs between-persona 0.9920; the own-average vs nearest-other distance margin is 0.0029 vs 0.0042. **Whitening (already mandatory in the architecture spec) is doing real work here** — the identity signal is well under 1% of the conditioning vector's magnitude.

---

### 3.3 Data status audit (2026-09-23; identity rows revised 2026-09-24)

**Currency**

| Dataset | Stream | Status |
|---|---|---|
| `hegre_corpus` | `pixel`, `auraface_lda`, `z_g`, `metadata` | ✅ current — rebuilt 2026-09-22/23, single basis, fingerprint `e2f66241288e1f50` |
| `ffhq/stratum` | `auraface_lda` | ✅ current — refit basis, norm 1.0, stamped `120e1c5a1dc4f423` — §2 |
| `ffhq/stratum` | `z_g` | ✅ same encoding as hegre (both postdate `encoder_production.npz`, 2026-06-23) |
| `ffhq/stratum` | `pixel`, `pose`, `dinov3_patches`, `t5_hidden`, `flux_latent`, `caption` | ✅ present |

**Completeness** — full directory scans, not samples:

- `hegre_corpus`: **31,711 / 31,711 complete** on all four streams, 0 missing, matches `_manifest.json` exactly.
- `ffhq/stratum`: 70,000 real samples + **1 `@eaDir`** (Synology metadata directory — must be excluded by any glob/loader).
  - `pixel`, `pose`, `dinov3_patches`, `t5_hidden`, `caption`, `flux_latent`: **70,000 / 70,000** ✅
  - `z_g.npy`: **139 missing (0.20%)**
  - `auraface_lda.npy`: **41 missing (0.06%)**
  - cross-check: the pooled LDA fit used exactly 69,960 FFHQ vectors — equal to the `auraface_lda` present count ✅

**⚠️ The remaining conditioning mismatch: `z_g`**

| | FFHQ (n=69,862) | hegre_corpus (n=31,711) |
|---|---|---|
| per-vector norm p50 | 6.39 | 8.49 |
| p95 | 10.10 | **27.36** |
| p99 | 13.27 | 42.57 |
| norm > 15 | 0.49% | **20.08%** |
| norm > 25 | 0.03% | **6.66%** |
| mean per-dim std | 1.138 | **1.916** |

`extract_zg_and_averages.py` documents norm > 25 as *"degenerate z_g (DWPose missed eyes/face → wild PCA projection)"* — but that filter is applied **only when computing persona averages**, not to the per-image vectors the corpus ships. So ~6.7% of `hegre_corpus` carries per-image `z_g` that the project's own heuristic calls degenerate, and the two datasets' geometry streams sit on visibly different scales.

Not settled whether this is genuine hegre domain/pose extremity or DWPose face-keypoint failure. Either way it must be resolved **consistently across all splits** so it does not become a split confound.

**No existing tool produces this verdict.** `experiments/geometry_pca/scripts/zg_full_corpus_audit.py` settles a *different* question — whether `z_g` encodes pose (`z_g→yaw` R²=0.98) or identity (AUC 0.90) versus the "pose-invariant by construction" claim in the standing docs. The degeneracy-threshold adjudication is new work.

Corpus `z_g` == the `zg/` source tree bit-exactly (‖diff‖ = 0.00000000 over 600 matched samples) — the corpus is a faithful copy; the issue is upstream.

### 3.4 Proposed split

Corpus structure (manifest + directory scan): **321 personas / 31,711 images**, capped at **100 images/persona** (min 19, median 100); **shoots (sets) per persona: min 1, median 5, max 42**.

| sets/persona | personas | images covered |
|---|---|---|
| ≥ 1 | 321 (100%) | 31,711 |
| ≥ 2 | 314 (97.8%) | 31,011 |
| ≥ 3 | 297 (92.5%) | 29,407 |
| ≥ 5 | 188 (58.6%) | 18,633 |
| ≥ 10 | 43 (13.4%) | 4,219 |

Existing convention: `prepare_cross_shoot_split(min_sets=2, seed=42)` in `geometry_pca/data_loader.py` holds out one *shoot* per persona as query, rest to index — that is the Phase 5b cross-shoot **evaluation** convention, not a training split.

Proposed three-way, persona-disjoint:

| split | personas | ≈ images | purpose |
|---|---|---|---|
| train | 249 (77.6%) | ≈24,600 | fit |
| val | 32 (10.0%) | ≈3,200 | model selection + in-loop instruments |
| **test (never touched)** | **40 (12.5%)** | **≈4,000** | opened once, at the end |

Rules:

1. **Persona-disjoint at every level.** Identity is the conditioning target; a persona in two splits measures memorization, not generalisation.
2. **Val must not be the training directory.** prx-tg's current `validation.stratum_dir` points at training data with `run_reconstruction: true` — that actively rewards copying (§5.2).
3. **Cross-shoot probe inside held-out personas.** For val/test personas with ≥2 shoots (314 available), hold out one shoot per persona to test *same person, different photo* — the one generalisation test FFHQ structurally cannot provide.
4. **Lock the split.** Record persona lists + a hash in `provenance.yaml`. Persona splits drift silently otherwise.
5. **Apply any `z_g` validity filter identically to all three splits**, so it cannot become a confound.
6. **FFHQ cannot be a val/test set for identity** (1 image/identity). If used at all: train-only regularizer with identity masked (Option B).

**Open decision:** the corpus caps at 100 images/persona — it uses 31,711 of the 166,204 approved hegre images (19%). The cap buys persona balance; lifting it buys pose coverage. Decide before the split is frozen, as it changes the split sizes.

---

## 4. Proposed gates

To be **pre-registered in prx-tg's ledger before each run**, with the gate text committed before results exist.

### 4.0 G0 — Photorealism (prx-tg's existing suite)
Unchanged; prx-tg owns this. Note it is currently unmet across arms (strike 1/3 on dip-conv-head, 1/3 on gamma2).

### 4.1 G1 — Identity binding (**the missing instrument**)
The Eidolon configs correctly set `run_dino_swap: false` / `run_text_manip: false` ("Eidolon has no DINO — skip", "no T5 text — skip"), **but no Eidolon-equivalent probe was added in their place.** This is the exact instrument-gap class that let the P1/PP taint run 10k steps unnoticed.

Proposed instrument — **identity swap**:
- render sample A with A's `auraface_lda`, then re-render with B's `auraface_lda`, everything else held fixed (same seed, same `z_g`);
- measure: rendered-image ΔLPIPS, and AuraFace cosine of the rendered faces against A and B;
- **gate:** identity-swapped renders must be significantly closer to the swapped-in identity than to the original (report the margin, and a null band from two renders at the same identity).

Also worth adding: **identity invariance across shoot** — two samples of the same persona with same `z_g` should render the same person.

### 4.2 G2 — Geometry binding (extend Arm O's gate)
Keep Arm O's `dim0 → yaw` monotonicity check at both 3000 and 5000 steps, and extend to more of the 27 morphology components (per the earlier Fisher-J split: 27 morphology / 23 transient). Same "zero collapse through the end of the run" clause.

### 4.3 G3 — Disentanglement / non-interference (**the core Eidolon claim**)
This is the hypothesis the whole project exists to test and is currently **not gated at all**:
- **identity swap must not move geometry** — measured pose delta (yaw/pitch/roll from DWPose on the render) below a pre-registered band;
- **geometry swap must not move identity** — measured identity delta below a pre-registered band;
- **gate:** both non-interference bands hold at the same checkpoint that passes G1 and G2.

### 4.4 G4 — Memorization ceiling (**the P1/PP lesson applied**)
- **never-trained persona:** hold out N personas entirely; render from their identity vectors; require the identity still binds (G1 margin holds on unseen identities).
- **never-trained image:** require reconstruction quality on *unseen* images of *seen* personas to be reported separately from that on seen images. **Memorization signature = unseen materially worse than seen.**
- **gate:** predefined maximum seen-vs-unseen recon gap.

### 4.5 G5 — Stability
No collapse over the full run **including** the CFG-dropped steps (the Arm N failure mode), with loss/gradient-norm telemetry reported — noting the gamma2 lesson that a healthy loss can coexist with luminance/contrast collapse.

---

## 5. Expectations / non-negotiables

1. **Instrument parity.** Every gate above must have code that runs during training, not a script run afterwards by hand. The P1/PP post-mortem is explicit that the two probes which would have caught it *existed and were disabled in config*.
2. **Train / val / never-touched-test split.** Currently `validation.stratum_dir` points at the *training* directory and `run_reconstruction: true` — i.e. validation reconstructs training images, which *rewards* memorization. This must change before the next Eidolon-relevant arm.
3. **No warm-start from the tainted lineage** (`pixel-posttrain`, `latent-first-pretrain`). Also do not warm-start the next Eidolon arm from an arm trained on the mixed-basis identity (§2) without re-establishing what its identity stream learned.
4. **Report CFG dropout marginals.** P1/PP's failure was partly a dropout-marginal problem (both CLS and text present only ~40%). For Eidolon: `p_uncond 0.10 / p_identity_only 0.20 / p_geometry_only 0.30` → **state the fraction of steps where both identity and geometry are present**, since that is the fraction where the disentanglement objective is actually trained.
5. **Report in Eidolon's units.** For anything Eidolon-facing: pixel output → AuraFace embedding → LDA coordinates, so identity claims are measured in the same space Eidolon ships.
6. **Pre-register first.** Gate text into the ledger before the run, evidence after.

---

## 6. Open questions for the Tim ↔ prx-tg discussion

1. **`z_g` encoding is compatible; its distribution is not.** Both datasets' `z_g` postdate `encoder_production.npz` (2026-06-23), so unlike identity this is **not** a staleness problem. But hegre's per-image `z_g` carries a 20% high-magnitude tail (6.7% past the project's own norm-25 degeneracy threshold) vs FFHQ's 0.5% / 0.03%, with mean per-dim std 1.92 vs 1.14. It needs a degeneracy verdict and a filter applied consistently across splits (§3.3). Separately: the standing docs call `z_g` "pose-invariant by construction" while a 2026-07-07 note measured `z_g→yaw` R²=0.98 — settle that contradiction before `z_g` is trusted as the geometry control.
2. **Who reprojects FFHQ?** Eidolon already has working reprojection machinery (`scripts/reproject_lda.py`, used to refresh 166,195 hegre per-image vectors). FFHQ raw AuraFace is on disk, so it is mechanical: `clean_auraface` + `project_to_lda` with the refit basis, over 70,001 samples. Needs an owner and a manifest/fingerprint so the same disease cannot recur.
3. **If FFHQ's identity is masked (Option B), does the 2.3 weight still make sense?** The weight was presumably chosen for identity diversity that Option B discards.
4. **Sequencing.** Is Eidolon's Phase 5 DiT work literally prx-tg's **Stage E**, or a parallel track? They need different data scopes (Stage E excludes hegre from P1/PP; hegre *is* Eidolon's data).
5. **Sapiens2 shape stream** — when does it enter the conditioning stack, and does it need its own gate (it is linearly AuraFace-orthogonal and had R²=−0.11 on the earlier probe)?

---

## 7. Evidence appendix

All commands run from `~/source/activity/eidolon` with `.venv/bin/python`. Identity
rows verified 2026-09-24; all other rows 2026-09-23.

| Claim | How verified |
|---|---|
| P1/PP use FFHQ stratum, differ only in latent vs pixel | `experiment-configs/{latent-first-pretrain,pixel-posttrain}/config.yaml` — `data.source: stratum`, `stratum_max_samples: 70000`; `in_channels` 16 vs 3 |
| Eidolon arms use `adapter.name: eidolon` + weighted FFHQ:hegre mix | `experiments/zg-token-basis-cfg-guard/config.yaml` L34–42, L94–101 |
| FFHQ has the full conditioning stack | `ls /mnt/nas-ai-models/training-data/ffhq/stratum/00000/` → `z_g.npy`, `auraface_lda.npy`, `pixel.npy`, `pose.npy`, `t5_hidden.npy`, `dinov3_patches.npy` |
| All identity dirs share one basis + convention | `hegre-dataset basis-fingerprint verify` → **OK on all three**, `120e1c5a1dc4f423` |
| Basis refit date | `experiments/geometry_pca/output/auraface_lda.npz` mtime 2026-07-23 16:11; backup `*.bak-20260720` |
| FFHQ identity matches hegre's convention | both norm `1.000000000`; FFHQ full scan 69,960/69,960 at unit norm, 0 degenerate, 0 NaN |
| Prior arms trained on two incompatible bases | FFHQ norm mean 0.352 vs hegre 1.000 → 2.841× (`scratch/check_stream_compat.py`) — the standing consequence in §2 |
| FFHQ ≈ 1 image per identity | `scratch/ffhq_identity_clusters.py` — 3,000 raw AuraFace vectors, 133 pairs > 0.6, 37 pairs > 0.7 |
| Persona-average identity index is sound | `scratch/test_average_discriminative.py` — R@1 0.8879, R@5 0.9502, R@10 0.9657 vs chance 0.0031 |
| Identity signal is compressed | persona averages pairwise cosine mean 0.9953; margin 0.0029 vs 0.0042 (`scratch/diagnose_identity_target.py`) |
| hegre corpus is single-basis & complete | `_manifest.json` fingerprint `e2f66241288e1f50`; on-disk == manifest, 31,711 samples |

Scratch scripts are at `~/.hermes/profiles/eidolon/cache/scratch/`.

---

## 8. Bottom line

- **P1/PP ≠ Eidolon arms.** They are a latent-vs-pixel training-mode spike on FFHQ stratum. Keep the two projects separate in the ledger language.
- **The actual Eidolon arms are the `exp/eidolon-conditioning` branch** — and their best result (Arm O, PASS) is a *geometry* result.
- **Their identity conditioning is unsound**: the mix blended FFHQ (pre-refit basis, per-image, norm 0.35) with hegre (refit basis, per-persona, norm 1.0). Verified bit-exact, not inferred.
- **Data is current and complete where it counts.** `hegre_corpus` is 31,711/31,711 complete on every stream, single-basis, manifest-verified. FFHQ is missing 139 `z_g` and 41 `auraface_lda` files and carries one `@eaDir` to exclude.
- **One conditioning stream is still mismatched: geometry.** Identity is now on a single basis and convention (§2); hegre's per-image `z_g` retains a degenerate tail and a different scale from FFHQ's (§3.3).
- **Recommended data: hegre_corpus only**, with a persona-disjoint 249 / 32 / 40 split, a cross-shoot probe inside held-out personas, and the split locked and hashed (§3.4).
- **Gates:** add the missing identity-binding probe (G1), gate the disentanglement claim (G3), and gate memorization with held-out personas and images (G4) — all pre-registered, all running in-training.