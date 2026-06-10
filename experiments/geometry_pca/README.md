# Geometry PCA Encoder (Phase 1)

This module builds the frozen $z_g$ (geometry) encoder for the Eidolon conditioning vector.

## Overview
It loads 68 facial keypoints from `stratum-ffhq`, applies Generalized Procrustes Analysis (GPA) to strip out translation, scale, and in-plane rotation, and fits a PCA model. The output is a highly compressed (k=50) set of orthogonal morphological sliders.

## Usage

1. **Install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Fit the Encoder:**
   ```bash
   # Fits PCA on 10,000 samples and saves to output/encoder.npz
   python scripts/01_fit_encoder.py --limit 10000 --k 50
   ```

3. **Generate Validation Plots:**
   ```bash
   # Generates scree, recon_error, and traversal plots in output/
   python scripts/02_validate.py
   ```
