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

### Preparing an Image Dataset (Example: FFHQ Thumbnails)

To train the encoder, you need a dataset of face images. As an example, here's how to download the FFHQ thumbnail dataset:

1.  **Clone the FFHQ repository:**
    Clone the official FFHQ dataset repository into a temporary location (e.g., outside of your `project-eidolon` directory to avoid nested git repositories).
    ```bash
    git clone https://github.com/NVlabs/ffhq-dataset.git
    cd ffhq-dataset
    ```

2.  **Download Thumbnails:**
    The FFHQ repository provides a script to download different parts of the dataset. We'll download only the thumbnails.
    First, create a directory where the downloaded images will be stored. It's often good practice to create this outside your project directory initially.
    ```bash
    # Example: From your project-eidolon directory's parent
    mkdir ../ffhq_data 
    ```
    Now, run the download script from within the cloned `ffhq-dataset` directory, pointing the output to your newly created folder:
    ```bash
    # Ensure you are inside the ffhq-dataset directory
    python download_ffhq.py --tasks thumbnails --outdir ../ffhq_data/thumbnails
    ```
    This script will download all 70,000 thumbnail images (128x128 pixels) and can take some time.

3.  **Organize Your Data:**
    Once the download is complete, you'll have a folder (e.g., `../ffhq_data/thumbnails`) full of images. For use with this project, it's recommended to move or copy these images into the `project-eidolon/data/` directory.
    ```bash
    # From your project-eidolon root directory:
    mkdir -p data/ffhq_thumbnails
    mv ../ffhq_data/thumbnails/* data/ffhq_thumbnails/
    ```
    Your images should now be located in `project-eidolon/data/ffhq_thumbnails/`. You can then set `IMAGE_DIR = "data/ffhq_thumbnails"` in `train.py`.

### Running the Training Script (Skeleton)

The `train.py` script is a skeleton for training the `EidolonEncoder`. To run it:

1.  Ensure you have completed all steps in the [Setup](#setup) section.
2.  Activate your virtual environment:
    ```bash
    source .venv/bin/activate
    ```
3.  **Crucially, update the `IMAGE_DIR` variable in `train.py`** to point to the directory containing your face image dataset (e.g., CelebA-HQ, FFHQ, or a custom dataset).
    ```python
    # In train.py, find and modify this line:
    IMAGE_DIR = "path/to/your/face/dataset" 
    ```
    For example, if you have images in `data/my_faces/`, change it to:
    ```python
    IMAGE_DIR = "data/my_faces"
    ```
    Ensure this directory contains `.png`, `.jpg`, or `.jpeg` image files.
4.  Run the script from the root directory of the project:
    ```bash
    python train.py
    ```
    This script will attempt to load data and start a training loop. Note that many parts of the forward pass and loss calculation are currently placeholders (marked with `TODO`) and will need to be implemented for meaningful training. The script will print basic progress information.

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
