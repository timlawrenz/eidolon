# Project Status — Eidolon

**Last updated:** 2026-09-24
**Phase / status:** Phase 5b concluded — between phases

## Current state

Phase 5a (text→identity Prior) and Phase 5b (Poser retrieval spike) are
concluded. The conditioning stack is settled: **DINOv3 masked patch tokens**
carry identity (Phase 4, AUC 0.797 cross-shoot), **z_g via DWPose** provides
identity-blind pose control (Phase 1-R), and **Sapiens2 dense keypoints** form
a complementary shape-morphology stream (linearly AuraFace-orthogonal, R²=−0.11).

**2026-07-20:** The Hegre dataset underwent massive cleaning (97k unreviewed
classified, ~54k bad_geometry reclassified). LDA basis refitted on 324 personas
(2.1×) and 166k images (3×). The refit raised GT-LDA ceiling from R@1=0.842
→ **0.854** — a tighter target for Phase 5 DiT fusion.

**2026-09-23:** Training corpus rebuilt on the refitted basis: **31,711 samples /
321 personas, 0 errors**, with `_manifest.json` recording the basis fingerprint
(`e2f66241288e1f50`). The prior corpus was revealed to be a *mixed-basis*
generation (37,011 dirs; 5,306 stale dirs carried pre-refit persona averages) —
retained at `hegre_corpus.old`. Corpus builds are now resumable
(`--skip-existing`) and auditable (manifest); per-image LDA reprojected onto the
new basis. Gate **G3 PASS**.

Dead partitions (z_d depth, z_a normals, DINO bridge) are permanently documented
and will not be re-attempted. No active training runs.

**2026-09-24:** **FFHQ identity reprojected onto the refit basis** — 69,960 files,
0 errors. `ffhq/stratum/{id}/auraface_lda.npy` was still on the *pre-refit* basis
(files 2026-06-30 vs refit 2026-07-23; proven bit-exact), so every `eidolon`-adapter
arm whose `stratum_dirs` included FFHQ fed a 64-d identity slot **two incompatible
encodings** (FFHQ norm 0.35 / hegre-corpus norm 1.0). FFHQ is now encoding-identical
to `hegre_corpus` (both norm `1.000000000`). A **basis fingerprint guard**
(`tools/hegre_dataset/basis_fingerprint.py`, `hegre-dataset basis-fingerprint
verify`) now lets loaders refuse mixed-basis input instead of silently training on
it; all three consumed dirs are stamped (`120e1c5a1dc4f423`). Arm
`ffhq-basis-reproject`, gate **G1–G4 PASS**.

⚠️ **Open consequence:** the identity conditioning of the five
`exp/eidolon-conditioning` arms in prx-tg is confounded by that mixed basis — no
identity-binding conclusion can be drawn from any of them, including Arm O's PASS
(whose *geometry* result stands independently). See
`docs/briefings/2026-09-23_prx-tg_eidolon-training-brief.md`.

✅ **Resolved 2026-09-24:** the `z_g` high-magnitude tail is **not** a degeneracy
artifact. `exp/zg-validity` (KILL of the belief) measured 60 samples at norm>25 vs 60
controls: **0 missing keypoints and 0 sub-0.3-confidence keypoints in either group**,
and all corpus samples are `approved` in the review DB — so the docstring's "DWPose
missed eyes/face" case is already handled upstream. The strata are separated by
**pose atypicality** (the high tail is atypical, often head-inverted poses with
*accurate* landmark alignment), not by landmark quality. `z_g` norm is a
pose-atypicality index, not a validity index — **no filter**, and the
`norm > 25 = degenerate` claim is retired
(`docs/DISCONTINUATION_NOTICE_zg_norm_filter.md`).
✅ **Resolved 2026-09-24 (b):** the `z_g` identity-content number is **re-measured and
corrected.** `exp/zg-identity-blindness` (verdict **PARTIAL**, but decisive on the
separation). The previously quoted **Fisher J = 0.059 is withdrawn** — it has **no
producing script on any branch**, so by this project's own rule it was not evidence,
and it is size-confounded. Re-measured on the curated corpus:

| stream | J | **J / noise floor** |
|---|---|---|
| per-image AuraFace-LDA | 2.0137 | **197.5×** |
| `z_g` (venue B) | **0.0847** | **8.31×** |

Raw J rose (0.059 → 0.085) but **size-corrected it fell (12.60× → 8.31×)** — the
floor `(C−1)/(N−C)` is 2.18× higher at 31,711 samples than at 69,110, so raw J rises
for free. The belief holds in direction (a **24× separation**); magnitude corrected.
Also established: the corpus **`auraface_lda.npy` is the persona centroid**
(321/321 personas bit-identical across all their samples, `S_W = 0` by construction)
— the centroid construction for the prx-tg arm is therefore already baked into the
data and verified, not assumed.

⚠️ **Still open:** (a) G3 — the retired claim drives a **persona-average filter** in
`extract_zg_and_averages.py`, so persona averages may be biased; (b) is the `z_g`
*encoding* reliable at pose-distribution extremes (reviewer's Sapiens-OOD
hypothesis); (c) extreme-pose/off-frame crops pass review and affect **all three
streams**, not just `z_g` — needs its own corpus-quality arm.

## Headline result so far

**AuraFace-LDA R@1 = 0.854 cross-shoot** (raised from 0.842 after dataset
cleaning + LDA basis refit). The retrieval space is sound; the gap is in the
text→LDA Prior (R@10=0.072, statistically indistinguishable from
random-projection null, p=0.063 at k=10).

## Immediate next action

**Phase 5: DiT Fusion Stack** — implement the 2-stream decoupled cross-attention
DiT with block-diagonal ingestion. Conditioning inputs settled:
- Identity: flesh-masked DINOv3 patch tokens
- Geometry control: z_g expanded tokens (DWPose, identity-blind)
- (Future) Shape-morphology: Sapiens2 stream (AuraFace-orthogonal)

Architecture reference: `docs/01_VISION_AND_ARCHITECTURE.md` §7.
No blockers. Ready to start.

## Active branches

| Branch | Workstream | Status |
|--------|-----------|--------|
| `exp/text-to-zg` | Phase 5a/b — Text→identity Prior + Poser retrieval | CONCLUDED |
| `exp/sapiens2-keypoints-study` | Sapiens2 keypoints — faithfulness + identity carrier | CONCLUDED |
| `exp/geometry-pca` | Phases 1–4 — geometry PCA, volumetric encoders, DINO bridge | CONCLUDED |
| `main` | Infrastructure, docs, tools | STABLE |
