# Project Eidolon is an attempt to describe a person, including the face, albedo, and body in a single vector.

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
    This project uses the FLAME model. When using the `.pkl` model files (like `flame2023.pkl`), the `chumpy` library is required. It will be installed as part of the command below.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download FLAME assets:**
    *   Go to [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/) and register for a free account.
    *   Download the following assets:
        *   FLAME 2023 (select the main "FLAME model" download, which typically includes `.pkl` files like `flame2023.pkl`).
        *   FLAME Vertex Masks (select "Vertex masks", e.g., `FLAME_masks.pkl` or `vertex_masks.npz`). The file `FLAME_masks.pkl` is present in your `data/flame_model` directory.
        *   FLAME Mediapipe Landmark Embedding (select "Landmark embedding for Mediapipe", e.g., `mediapipe_landmark_embedding.npz`). This is present in your `data/flame_model` directory.
    *   Unzip the downloaded files.
    *   Create a directory `data/flame_model/` in the root of this project (if it doesn't exist).
    *   Place the unzipped model files into the `data/flame_model/` directory. Based on your files, you should have at least `flame2023.pkl`, `FLAME_masks.pkl`, and `mediapipe_landmark_embedding.npz`. The `head_template_mesh.obj` is also useful if provided in the FLAME download.

## Usage

To run the main script, which loads the FLAME model, renders an average face, and performs a basic test of the `EidolonEncoder` (downloading ResNet-50 weights if not already cached):

1.  Ensure you have completed all steps in the [Setup](#setup) section.
2.  Activate your virtual environment:
    ```bash
    source .venv/bin/activate
    ```
3.  Run the script from the root directory of the project:
    ```bash
    python main.py
    ```
    The first time you run this, it may take a few moments to download the pre-trained ResNet-50 model weights. A plot window showing the rendered average face should appear.

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
