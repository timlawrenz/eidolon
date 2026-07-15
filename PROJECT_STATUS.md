# Project Status — Eidolon

**Last updated:** 2026-07-15
**Phase / status:** Phase 5b concluded — between phases

## Current state

Phase 5a (text→identity Prior) and Phase 5b (Poser retrieval spike) are
concluded. The conditioning stack is settled: **DINOv3 masked patch tokens**
carry identity (Phase 4, AUC 0.797 cross-shoot), **z_g via DWPose** provides
identity-blind pose control (Phase 1-R), and **Sapiens2 dense keypoints** form
a complementary shape-morphology stream (linearly AuraFace-orthogonal, R²=−0.11).

Dead partitions (z_d depth, z_a normals, DINO bridge) are permanently documented
and will not be re-attempted. No active training runs.

## Headline result so far

**AuraFace-LDA R@1 = 0.842 cross-shoot** — first proof that AuraFace-LDA
is a genuine cross-shoot identity carrier (Phase 5b GT-LDA ceiling gate).
The retrieval space is sound; the gap is in the text→LDA Prior (R@10=0.072,
statistically indistinguishable from random-projection null, p=0.063 at k=10).

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
