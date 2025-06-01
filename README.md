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
        *   FLAME 2023 (select "FLAME 2023 ... no chumpy dependency"). This will provide `.npz` files (e.g., `generic_model.npz`, `female_model.npz`, `male_model.npz`).
        *   FLAME Vertex Masks (select "Vertex masks", e.g., `vertex_masks.npz`).
        *   FLAME Mediapipe Landmark Embedding (select "Landmark embedding for Mediapipe", e.g., `mediapipe_landmark_embedding.npz`).
    *   Unzip the downloaded files.
    *   Create a directory `data/flame_model/` in the root of this project.
    *   Place the unzipped model files into the `data/flame_model/` directory. You should have at least `generic_model.npz` (or `female_model.npz`/`male_model.npz`), `head_template_mesh.obj` (often included or can be generated), `mediapipe_landmark_embedding.npz`, and `vertex_masks.npz` in this directory.

## Usage

(Add instructions on how to run your project here)

## Running the Notebooks

1.  Ensure you have completed all steps in the [Setup](#setup) section, including installing dependencies and downloading FLAME assets.
2.  Activate your virtual environment:
    ```bash
    source .venv/bin/activate
    ```
3.  Start JupyterLab from the root directory of the project:
    ```bash
    jupyter lab
    ```
4.  JupyterLab will open in your web browser. Navigate to the `notebooks/` directory and open `01_exploration_and_rendering.ipynb`.
5.  Run the cells in the notebook sequentially.

## References

(Add instructions on how to run your project here)

## References

This project utilizes the FLAME model:

```bibtex
@article{FLAME:SiggraphAsia2017, 
  title = {Learning a model of facial shape and expression from {4D} scans}, 
  author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier}, 
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)}, 
  volume = {36}, 
  number = {6}, 
  year = {2017}, 
  pages = {194:1--194:17},
  url = {https://doi.org/10.1145/3130800.3130813} 
}
```
