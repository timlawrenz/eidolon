---
tags:
  - "#project/eidolon"
  - "#metadata/project/v1"
repo: /home/tim/source/activity/eidolon
type: private
---
# Eidolon Literature Synthesis

## 🔬 Papers Relevant to Project Eidolon

*Searched arXiv + Semantic Scholar + web across: disentangled representations, PCA orthogonality, decoupled cross-attention, DiT face generation, 3DMM lineage, and Sapiens.*

### 🧱 Pillar 1: PCA-Guaranteed Orthogonality & Disentanglement

These speak directly to Eidolon's core principle — PCA-enforced orthogonality replacing entangled latent spaces.

- **[[00 - articles/ShapeFusion A 3D diffusion model for localized shape editing|ShapeFusion: A 3D Diffusion Model for Localized Shape Editing]]** *Potamias et al. (Zafeiriou group)* (2024) — **Most aligned.** Uses PCA for orthonormal shape decomposition, then a diffusion model for localized editing. Directly confronts the trade-off between PCA's global nature and local controllability — the exact tension Eidolon's block-diagonal firewall addresses. Has a project page with code.
- **[[00 - articles/PCA-VAE Differentiable Subspace Quantization without Codebook Collapse|PCA-VAE: Differentiable Subspace Quantization without Codebook Collapse]]** *Lu et al.* (2026) — Replaces VQ with PCA-based quantization — fully differentiable, no codebook collapse. Demonstrates PCA as a principled alternative to learned codebooks.
- **[[00 - articles/Variational Autoencoders Pursue PCA Directions (by Accident)|VAEs Pursue PCA Directions (by Accident)]]** *Rolinek et al.* (2019) — **Theoretical foundation.** Proves that under certain conditions, VAE latent axes converge to PCA directions. Explains *why* orthogonal decompositions emerge naturally.
- **[[00 - articles/NeurIPS Poster Can Diffusion Models Disentangle A Theoretical Perspective|Can Diffusion Models Disentangle? A Theoretical Perspective]]** *(NeurIPS 2025)* (2025) — First theoretical framework for disentanglement in diffusion models. Addresses whether the diffusion process itself provides disentanglement guarantees.
- **[[00 - articles/NeurIPS Poster Diffusion Model with Cross Attention as an Inductive Bias for Disentanglement|Diffusion Model with Cross Attention as an Inductive Bias for Disentanglement]]** *(NeurIPS 2024)* (2024) — Shows cross-attention + diffusion bottlenecks naturally induce disentanglement — **no regularization needed**. Supports Eidolon's decoupled cross-attention design as intrinsically disentanglement-promoting.

### 🔒 Pillar 2: Block-Diagonal / Decoupled Conditioning

These validate Eidolon's architectural choice: independent per-modality ingestion + decoupled cross-attention.

- **[[00 - articles/IP-Adapter Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models|IP-Adapter: Text Compatible Image Prompt Adapter]]** *Ye et al.* (2023) — **The canonical reference.** Introduces decoupled cross-attention (separate K/V per condition, summed outputs). This is the exact pattern Eidolon uses for its three-modality firewall.
- **[[00 - articles/Multivariate Diffusion Transformer with Decoupled Attention for High-Fidelity Mask-Text Collaborative Facial Generation|MDiTFace: Multivariate DiT with Decoupled Attention for Mask-Text Facial Generation]]** *Cao et al.* (2025) — Custom DiT with **decoupled attention** across mask and text modalities. Unified tokenization → separate attention heads → summed fusion. Close architectural cousin.
- **[[00 - articles/DCMorph Face Morphing via Dual-Stream Cross-Attention Diffusion|DCMorph: Face Morphing via Dual-Stream Cross-Attention Diffusion]]** *Chettaoui et al.* (2026) — Dual-stream identity-conditioned latent diffusion. Operates at both identity conditioning and latent levels simultaneously — validates multi-stream conditioning efficacy.
- **[[00 - articles/IP-Adapter Is All You Need Towards Fine-Tuning-Free Diffusion-Based Talking Face Generation|IP-Adapter Is All You Need: Tuning-Free Diffusion-Based Talking Face Generation]]** *Wu et al.* (2026) — Shows IP-Adapter alone (no fine-tuning) suffices for high-quality talking face generation. Reinforces the power of decoupled conditioning.

### 🎨 Pillar 3: DiT Architectures for Face Generation

Competitor approaches using Diffusion Transformers for face synthesis with conditioning.

- **[[00 - articles/MMFace-DiT A Dual-Stream Diffusion Transformer for High-Fidelity Multimodal Face Generation|MMFace-DiT: A Dual-Stream DiT for Multimodal Face Generation]]** *Krishnamurthy & Rattani (CVPR 2026)* (2026) — Dual-stream DiT fusing text + spatial priors (seg masks, sketches, edge maps). CVPR 2026, has code. Most architecturally similar to Eidolon's DiT plans.
- **[[00 - articles/Mixture of Global and Local Experts with Diffusion Transformer for Controllable Face Generation|Face-MoGLE: Mixture of Global and Local Experts with DiT]]** *Zou et al.* (2025) — DiT with expert specialization: semantic-decoupled latent modeling via mask-conditioned routing. Addresses disentanglement at the architecture level.
- **[[00 - articles/Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control|Face-Adapter: Fine-Grained ID and Attribute Control for Diffusion]]** *Han et al.* (2024) — Adapter-based fine-grained control over identity and attributes in pre-trained diffusion models. Relevant for how to inject disentangled controls without full retraining.
- **[[00 - articles/OminiControl2 Efficient Conditioning for Diffusion Transformers|OminiControl2: Efficient Conditioning for DiT]]** *Tan et al.* (2025) — Lightweight conditioning for DiT models — relevant for making the Eidolon DiT efficient.

### 🏛️ Pillar 4: 3DMM / Statistical Shape Model Lineage

The geometric half of Eidolon — from classical PCA-based 3D Morphable Models to neural variants.

- **[[00 - articles/GIF Generative Interpretable Faces|GIF: Generative Interpretable Faces]]** *Ghosh et al. (Michael Black group)* (2020) — FLAME parametric model + StyleGAN renderer. Key reference for controlling a generative model with a 3DMM parameter vector.
- **[[00 - articles/Pixel3DMM Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction|Pixel3DMM: Versatile Screen-Space Priors for 3D Face Reconstruction]]** *Giebenhain et al.* (2025) — Uses DINO foundation features + ViT heads to predict per-pixel geometric cues for 3DMM fitting. State-of-the-art single-image 3D reconstruction.
- **[[00 - articles/Text2Face A Multi-Modal 3D Face Model|Text2Face: Multi-Modal 3D Face Model]]** *Rowan et al.* (2023) — Extends FLAME to a text+image joint latent space — first 3DMM controllable via text prompts.
- **[[00 - articles/KaoLRM Repurposing Pre-trained Large Reconstruction Models for Parametric 3D Face Reconstruction|KaoLRM: Repurposing Large Reconstruction Models for Parametric 3D Face Reconstruction]]** *Zhu et al.* (2026) — Retargets LRM (Large Reconstruction Model) priors for 3DMM face reconstruction from single views.

### 🔧 Pillar 5: Data Pipeline (Sapiens)

- **[[00 - articles/Sapiens Foundation for Human Vision Models|Sapiens: Foundation for Human Vision Models]]** *Khirodkar et al. (Meta)* (2024) — **This is Eidolon's data backbone.** Provides pose, segmentation, depth, and surface normals on 300M+ human images at 1K resolution. The stratum-ffhq depth/normal/pose/seg layers come from Sapiens.

### 📊 Cross-Cutting: Disentanglement Evaluation

- **[[00 - articles/Evaluating Disentanglement in Generative Models Without Knowledge of Latent Factors|Evaluating Disentanglement in Generative Models Without Knowledge of Latent Factors]]** *Holtz et al.* (2022) — Unsupervised disentanglement metric — no ground truth needed. Could serve as an Eidolon validation gate.
- **[[00 - articles/Disentanglement via Latent Quantization|Disentanglement via Latent Quantization]]** *Hsu et al.* (2023) — Shows quantization itself induces disentanglement. Relevant for any discrete-bottleneck variant.

### 🎯 Summary

The search confirms that **Eidolon sits at a novel intersection** no single paper covers:

1. **PCA orthogonality** + **block-diagonal ingestion** + **decoupled cross-attention** + **DiT generator** — each component has literature support, but the combination is unique
2. **ShapeFusion** is the closest paper (PCA for 3D shape + diffusion), but it operates on mesh vertices, not a unified geometry/depth/albedo latent
3. **MMFace-DiT** is the closest DiT competitor but uses standard fusion, not the block-diagonal firewall
4. **IP-Adapter** provides the decoupled attention pattern, but was designed for image prompting, not multi-modal biometric latents
5. The **NeurIPS 2024/2025 disentanglement theory papers** provide theoretical backing that the architecture *itself* (cross-attention + diffusion bottlenecks) promotes disentanglement — validating the design

---

## 📖 ShapeFusion Deep Dive & Reference Trace

### What ShapeFusion Does

**Core idea**: PCA-based 3DMMs have *entangled, global* latent spaces — moving one PCA coefficient affects the entire mesh. ShapeFusion solves this by **masked diffusion**: inject noise only into a user-selected geodesic region, then denoise it. The un-noised anchor point + masked region = inherently localized editing *by construction*, not by latent disentanglement.

**Key architectural choices**:
- Hierarchical mesh convolutions (3 resolution levels) for long-range smoothness
- Learnable vertex-index positional encoding (breaks permutation equivariance for topology-specific priors)
- Spiral mesh convolutions (CoMA-style, from Bouritsas et al.)
- Trained on UHM (faces), STAR (bodies), MimicMe (real 4D facial scans)

### How ShapeFusion Relates to Eidolon

| ShapeFusion | Eidolon |
|---|---|
| Operates on **3D mesh vertices** (x,y,z coordinates) | Operates on **2D images** via DiT |
| PCA is the *problem* (too global) | PCA is the *solution* (guarantees orthogonality) |
| Solves localization via **masked diffusion** on mesh | Solves localization via **block-diagonal firewall** in DiT |
| Anchor point + geodesic region = local control | Per-modality experts + decoupled cross-attention = disentangled control |
| Diffusion on vertex coordinates | Diffusion in latent image space |

**The papers are complementary, not competing.** ShapeFusion shows how to get localized 3D control despite PCA's global nature; Eidolon shows how to preserve PCA orthogonality through a generative pipeline. A future synthesis could use ShapeFusion's masked approach on the *geometry branch* of Eidolon's latent vector.

---

### 🔗 Reference Graph: Key Papers to Follow

Here are the most relevant references from ShapeFusion's bibliography, organized by their connection to Eidolon:

#### Tier 1 — Directly Relevant to Eidolon's Architecture

- [[00 - articles/3D Generative Model Latent Disentanglement via Local Eigenprojection|Foti et al. 2023 — 3D Generative Model Latent Disentanglement via Local Eigenprojection]]: Uses **spectral geometry** (eigenprojection on mesh Laplacian) to disentangle 3D shape latent spaces. Directly relevant — this is the mathematical cousin of Eidolon's PCA approach, but operating on mesh spectra rather than coordinate PCA. The loss function is grounded in differential geometry.
- [[00 - articles/Neural 3D Morphable Models Spiral Convolutional Networks for 3D Shape Representation Learning and Generation|Bouritsas et al. 2019 — Neural 3D Morphable Models: Spiral Convolutional Networks]]: The neural alternative to PCA-based 3DMMs. Uses spiral convolutions on mesh topology. This is what Eidolon is deliberately *not* doing (Eidolon chooses PCA over learned latents for orthogonality guarantees), but it's essential to understand the trade-off.
- [[00 - articles/Locally Adaptive Neural 3D Morphable Models|Tarasiou et al. 2024 — Locally Adaptive Neural 3D Morphable Models (LAMM)]]: Very recent (Jan 2024). Self-supervised AE where sparse control vertices overwrite encoded geometry. Directly addresses the *localized* vs *global* tension — same problem Eidolon's block-diagonal firewall tackles, but on meshes. From the same group as ShapeFusion.
- [[00 - articles/MeshDiffusion Score-based Generative 3D Mesh Modeling|Liu et al. 2023 — MeshDiffusion: Score-based Generative 3D Mesh Modeling]]: Applies DDPM to mesh generation with fixed topology. The diffusion-on-structured-geometry paradigm that ShapeFusion inherits. Relevant for understanding how diffusion can be applied to structured (non-image) data.

#### Tier 2 — Foundational 3D Face/Body Models

- [[00 - articles/FLAME|Li et al. 2017 — FLAME: Learning a Model of Facial Shape and Expression from 4D Scans]] (SIGGRAPH Asia 2017): **The canonical 3D face model.** Low-dimensional PCA-based shape + expression space from 4D scans. This is the direct lineage of Eidolon's geometry branch. FLAME uses PCA for disentangled identity/expression — the same principle Eidolon extends to geometry/depth/albedo.
- [[00 - articles/SMPL2015.pdf|Loper et al. 2015 — SMPL: A Skinned Multi-Person Linear Model]] (SIGGRAPH Asia 2015): The body-model equivalent of FLAME. PCA-based shape space + pose blendshapes. Foundational reference for the PCA approach to human shape.
- [[00 - articles/CVPR 2016 Open Access Repository|Booth et al. 2016 — A 3D Morphable Model Learnt from 10,000 Faces (LSFM)]] (CVPR 2016): The large-scale PCA face model from the Zafeiriou group. The "big PCA works" evidence.
- [[00 - articles/Towards a complete 3D morphable model of the human head|Ploumpis et al. 2020 — Towards a Complete 3D Morphable Model of the Human Head]] (TPAMI 2020): Extends 3DMM to the full head (face + cranium). From the same group.

#### Tier 3 — Disentanglement Theory & Methods

- [[00 - articles/3D Shape Variational Autoencoder Latent Disentanglement via Mini-Batch Feature Swapping for Bodies and Faces|Foti et al. 2022 — 3D Shape VAE Latent Disentanglement via Mini-Batch Feature Swapping]]: Self-supervised disentanglement for 3D shape VAEs. Swaps features between samples to enforce independence. An alternative to PCA-based disentanglement worth comparing against.
- [[00 - articles/[PDF] Unsupervised Shape and Pose Disentanglement for 3D Meshes Semantic Scholar|Zhou et al. 2020 — Unsupervised Shape and Pose Disentanglement for 3D Meshes]] (ECCV 2020): Unsupervised disentanglement of shape and pose on 3D meshes. Directly relevant to the z_g vs z_d separation in Eidolon.
- [[00 - articles/Generating 3D faces using Convolutional Mesh Autoencoders|Ranjan et al. 2018 — Generating 3D Faces Using Convolutional Mesh Autoencoders (CoMA)]] (ECCV 2018): The mesh autoencoder paper that introduced the spiral convolution used by ShapeFusion. Neural face generation on meshes.

---

### 🎯 Key Takeaways for Eidolon

1. **ShapeFusion confirms PCA's limitation** (global, not local) — but Eidolon's block-diagonal approach is a *different* solution to the same problem. ShapeFusion uses masked diffusion on meshes; Eidolon uses architectural firewalls in DiT. Both are valid.
2. **Foti's eigenprojection work** is the closest mathematical parallel — spectral geometry for disentanglement. Worth reading in full to see if the mesh Laplacian eigenprojection can inform Eidolon's PCA pipeline (e.g., as a validation metric).
3. **LAMM** is a direct peer from the same group — published 3 months before ShapeFusion, using control vertices for local editing. The rapid iteration in this group (Neural 3DMM → LAMM → ShapeFusion in 5 years) shows this is an active, unsolved problem.
4. **FLAME is essential context** — every 3D face paper cites it. Understanding FLAME's PCA-based identity/expression disentanglement would ground Eidolon's approach in the established literature.
5. **The gap Eidolon fills**: none of these papers combine (a) PCA orthogonality guarantees, (b) block-diagonal architectural firewalls, (c) decoupled cross-attention, and (d) a DiT image generator into one system. Each component exists separately.

---

## 🔬 Research Synthesis (2026-06-10)

*Cross-referenced 31 linked papers from the project log. All present in the vault under `00 - articles/`.*

### 1. Decoupled Cross-Attention is Theoretically Sound
- **IP-Adapter (Ye et al., 2023):** Decoupled K/V per modality, summed outputs. Only 22M parameters matches full fine-tuning.
- **NeurIPS 2024 — Cross-Attention as Inductive Bias:** Cross-attention + diffusion bottlenecks naturally induce disentanglement — no regularization needed.
**→ Implication for Eidolon:** The block-diagonal firewall is not just an engineering choice; the architecture itself promotes disentanglement.

### 2. PCA Orthogonality: The Theoretical Case
- **Rolinek et al. (2019):** VAE latent axes converge to PCA directions. This is why orthogonal decompositions emerge naturally.
- **PCA-VAE (Lu et al., 2026):** PCA-based quantization: fully differentiable.
- **FLAME / LSFM:** 30 years of 3DMM research validates PCA for human shape.
**→ Implication for Eidolon:** PCA-guaranteed orthogonality has theoretical justification and a 30-year track record.

### 3. Competitor Landscape: The Gap Remains Open
- **MMFace-DiT (CVPR 2026):** Dual-stream DiT. Eidolon differs: 3 modalities (not 2), PCA-guaranteed orthogonality, block-diagonal.
- **MDiTFace (Cao et al., 2025):** Decoupled attention across mask + text. Eidolon differs: decoupling at ingestion stage.
- **Face-MoGLE (Zou et al., 2025):** DiT with mask-conditioned routing. Eidolon differs: explicit architectural firewalls.
- **ShapeFusion (Potamias et al., 2024):** PCA is the problem; solves via masked diffusion on meshes. Eidolon differs: PCA is the solution; solves via block-diagonal firewalls in DiT.
**→ Confirmed:** No paper combines PCA orthogonality + block-diagonal firewalls + decoupled cross-attention + DiT generator.

### 4. Disentanglement Can Be Measured (Validation Gate)
- **Holtz et al. (2022) / Hsu et al. (2023):** Unsupervised disentanglement metric.
**→ Implication:** These provide a principled validation gate. Measure disentanglement on real images using Holtz's metric.

### 5. Conditioning Efficiency Matters
- **OminiControl2 / Face-Adapter:** Efficient conditioning for DiT / fine-grained ID control.
**→ Implication:** Eidolon's `z = [z_g | z_d | z_a]` concatenation creates a longer conditioning vector; OminiControl2's approach is directly relevant.

### 6. Data Pipeline Validation
- **Sapiens (Meta, 2024) / Pixel3DMM (2025):** 2D pose, segmentation, depth on 300M+ images.
**→ Implication:** The stratum-hq → Sapiens pipeline for z_d is validated.

### 7. Critical Assessment
**What the literature does NOT support:**
- That learned latent spaces are better than PCA for orthogonality (Rolinek shows convergence to PCA).
- That multi-modal fusion requires joint training (IP-Adapter proves decoupled matches fine-tuning).
- That disentanglement requires complex regularization (NeurIPS 2024 shows architecture alone suffices).

**What remains unproven (needs empirical verification):**
- Whether PCA on 2D image features preserves meaningful geometric disentanglement.
- Whether the block-diagonal firewall prevents information leakage in practice.
- Whether 3-modality ingestion (z_g, z_d, z_a) is sufficient.

**Bottom line:** The literature survey confirms Eidolon's architectural choices are well-grounded. The combination of PCA orthogonality + block-diagonal ingestion + decoupled cross-attention + DiT generator remains unique.