# Phase 5a: Text-to-Identity Priors — Implementation Plan

> **For Hermes:** Execute sequentially with TDD (RED-GREEN-REFACTOR); commit after each phase.
**Goal:** Train two Rectified Flow Matching Priors: text→z_g (50-d geometry) and text→AuraFace-LDA (64-d identity).
**Architecture:** AdaLN-ResNet MLPs (~12 residual blocks, hidden ~1024–2048) with T5 text conditioning via adaptive LayerNorm. Rectified Flow Matching (velocity prediction, ~10 Euler steps at inference).
**Tech Stack:** NumPy + PyTorch, T5 embeddings from Stratum-HQ, NAS data paths, CPU-bound (Strix Halo 128GB).
**Branch:** `exp/text-to-zg`
**Pre-registration:** `docs/02_EXPERIMENTS_AND_RESULTS.md` §[PRE-REGISTERED] Phase 5a

---

### Phase A: Permanent LDA Basis + project_to_lda()

**Objective:** Move the LDA basis from `/tmp/` to permanent storage and expose a reusable `project_to_lda()` function in the preprocessing module.

**Files:** Create `experiments/geometry_pca/output/auraface_lda.npz`, modify `geometry_pca/auraface_preprocessing.py`, create test.

#### Task A1: Move LDA artifacts to permanent storage
Copy `/tmp/lda_W.npy` (basis), `/tmp/lda_evals.npy` (eigenvalues) to `output/auraface_lda.npz` with descriptive keys.
Add mean vector (`mu_pool`) and PC1 for consistency with preprocessing stack.

#### Task A2: Write failing test for project_to_lda()
Create `tests/test_auraface_preprocessing.py` with:
- `test_project_to_lda_shape`: single vector (512,) → (64,), batch (N,512) → (N,64)
- `test_project_to_lda_reconstruct_cycle`: project→reconstruct via `W @ coords + mu` → verify shapes
- `test_project_to_lda_preserves_preprocessing_order`: apply clean_auraface then project_to_lda → verify yaw+PC1 projections are zero BEFORE LDA projection

#### Task A3: Implement project_to_lda() and add_to_lda_target()
In `geometry_pca/auraface_preprocessing.py`:
```python
def project_to_lda(v_clean):
    """Project cleaned (512-d) AuraFace vector onto the LDA identity basis.
    Args: v_clean — (512,) or (N,512) from clean_auraface()
    Returns: (64,) or (N,64) LDA coordinates"""
    
def lda_to_full(coords):
    """Reconstruct full 512-d vector from LDA coordinates.
    Returns: (512,) or (N,512) — still needs L2-normalize before DiT use."""
```

#### Task A4: Run tests → GREEN → commit
```bash
pytest tests/test_auraface_preprocessing.py -v
git add ... && git commit -m "feat: add project_to_lda() to auraface_preprocessing"
```

---

### Phase B: Prior 1 — text→z_g Flow Matching MLP (50-d)

**Objective:** Build a Rectified Flow Matching MLP that maps T5 text embeddings → 50-d z_g vectors. Gated by G1 (held-out MSE ratio < 1.0).

**Files:** Create `experiments/geometry_pca/priors/` package (`__init__.py`, `flow_matching.py`, `models.py`, `data.py`), create `tests/test_flow_matching.py`

#### Task B1: Write failing test — Flow Matching on synthetic 2D data
Create `tests/test_flow_matching.py`:
- `test_flow_matching_synthetic_2d`: generate 8 Gaussians in 2D, train 2-layer MLP with RFM for 200 steps, verify MSE < 0.05 on held-out
- `test_velocity_prediction_shape`: verify model outputs (N, d) shape matching noise input
- `test_inference_multistep`: sample noise, run 10 Euler steps, verify output is finite and plausible

#### Task B2: Implement flow_matching.py
```python
class RectifiedFlowMatching:
    def __init__(self, model, d_output, n_steps=10, sigma_min=1e-4):
    def loss(self, x_1, cond):  # samples t∈[0,1], x_t = (1-t)*x_0 + t*x_1, returns MSE(v_pred, x_1-x_0)
    def sample(self, cond, n_samples=1):  # Euler ODE solver, 10 steps
```
Model: `x_0 ~ N(0,I)`, `v_target = x_1 - x_0`, straight-line interpolation.

#### Task B3: Write failing test — AdaLN-ResNet model
- `test_resblock_shape`: (B, 1+d_model+cond) → (B, d_model)
- `test_adaln_resnet_forward`: (B, 50) noise + (B, 4096) T5 → (B, 50) velocity
- `test_conditional_on_text`: different T5 embeddings → different velocity outputs

#### Task B4: Implement models.py
AdaLN-ResNet: ~12 ResBlocks, each with linear→SiLU→linear conditioning projection that computes scale/shift for LayerNorm. Hidden dim ~1024–2048.

#### Task B5: Write failing test — data loading
- `test_ffhq_data_loader`: load T5 (.npy) and z_g (.npy) for first 100 FFHQ samples, verify paired shapes
- `test_train_val_split`: held-out identities, no same-identity leakage

#### Task B6: Implement data.py
Load paired (T5, z_g) from NAS paths for FFHQ. Optional: Hegre centroids.

#### Task B7: Integration test — end-to-end training on FFHQ subset
Train on 1000 FFHQ samples for 500 steps, verify loss decreases monotonically, save checkpoint, test sampling.

#### Task B8: RED→GREEN→commit for each sub-task

---

### Phase C: Prior 2 — text→AuraFace-LDA Flow Matching (64-d)

**Objective:** Build a Flow Matching MLP for AuraFace-LDA coordinates. Same architecture, cosine-based evaluation. Gated by G2 (held-out cosine > 0.3).

**Files:** Extend `priors/` package, add to `tests/test_flow_matching.py`

#### Task C1: Write failing test — AuraFace-LDA data loading
- `test_ffhq_lda_data`: load T5, AuraFace, apply clean_auraface+project_to_lda → verify (N,64) targets

#### Task C2: Write failing test — cosine-aware flow
- `test_lda_noise_distribution`: noise sampled in tangent space of hypersphere?
- `test_lda_cosine_gate`: train on 1000 samples, verify held-out cosine > 0.1 initially

#### Task C3: Implement Prior 2 training
Reuse Flow Matching + AdaLN-ResNet from Phase B. Adjust output dim to 64. Add L2-normalization of reconstructed output before cosine evaluation.

#### Task C4: RED→GREEN→commit

---

### Phase D: Gate Evaluation

**Objective:** Run G1 and G2 on held-out identities. Record verdicts in ledger.

#### Task D1: Compute σ²_w (within-person z_g variance)
From Hegre persona clusters (323 identities, excluding `hera`). Save to data file.

#### Task D2: Run G1 — z_g Prior gate
```bash
python scripts/pipeline/evaluate_gates.py --prior z_g --split held_out_ids
```
Expected output: MSE ratio, whether < 1.0, bootstrap CI.

#### Task D3: Run G2 — AuraFace-LDA Prior gate
```bash
python scripts/pipeline/evaluate_gates.py --prior lda --split held_out_ids
```
Expected output: mean cosine, whether > 0.3, bootstrap CI.

#### Task D4: Record verdicts in ledger
Update `docs/02_EXPERIMENTS_AND_RESULTS.md` Phase 5a entry: change `[PRE-REGISTERED]` → `[CONCLUDED — PASS/FAIL]`, append gate numbers.

#### Task D5: Commit gate results + updated ledger

---

### Phase E (Optional): Joint Prior Ablation

Only run if G1+G2 both pass. Train single Prior predicting [z_g | LDA] concatenated (114-d). Compare G3 results to G1+G2.

---

### Verification Checklist
- [ ] Pre-registration entry in ledger before any code
- [ ] All tests RED before GREEN (TDD)
- [ ] Invariant tests (not change-detector snapshots)
- [ ] All data paths use NAS (no local caching)
- [ ] Git commits after each phase
- [ ] Gates evaluated on held-out identities only
- [ ] Bootstrap CIs on all gate numbers
- [ ] Ledger updated with final verdicts
