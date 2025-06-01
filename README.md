# Project Eidolon

This project requires Python 3.10.
It is currently configured to use CUDA 12.1 due to the PyTorch and PyTorch3D dependencies specified in `requirements.txt`. If you have a different CUDA version or require CPU-only support, you may need to adjust the `--find-links` URL and relevant package versions in `requirements.txt`. Refer to the PyTorch and PyTorch3D installation guides for alternative options.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python3.10 -m venv .venv
    ```

2.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download FLAME assets:**
    *   Go to [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/) and register for a free account.
    *   Download the following assets:
        *   FLAME 2020 (select "FLAME model (female, male, gender-neutral)")
        *   FLAME Vertex Masks (select "Vertex masks")
        *   FLAME Mediapipe Landmark Embedding (select "Landmark embedding for Mediapipe")
    *   Unzip the downloaded files.
    *   Create a directory `data/flame_model/` in the root of this project.
    *   Place the unzipped model files into the `data/flame_model/` directory. You should have at least `generic_model.pkl`, `head_template_mesh.obj`, `mediapipe_landmark_embedding.npz`, and `vertex_masks.npz` (or similar, depending on the exact files from the FLAME 2020 download) in this directory.

## Usage

(Add instructions on how to run your project here)
