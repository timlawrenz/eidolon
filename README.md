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

### Preparing an Image Dataset (Example: FFHQ Thumbnails via Hugging Face)

To train the encoder, you need a dataset of face images. A reliable way to get the FFHQ dataset is through Hugging Face's `datasets` library. This method downloads the high-resolution images and then resizes them to create thumbnails.

1.  **Install the `datasets` library:**
    If you haven't already, ensure your virtual environment is active and install the library:
    ```bash
    pip install datasets
    ```
    This dependency has also been added to `requirements.txt`.

2.  **Run the Download and Processing Script:**
    A script `scripts/download_ffhq_huggingface.py` is provided to download the FFHQ dataset, resize images to 128x128 thumbnails, and save them to `data/ffhq_thumbnails_128/`.

    First, ensure the `scripts` directory exists. Then, run the script from the root of your `project-eidolon` directory:
    ```bash
    python scripts/download_ffhq_huggingface.py
    ```
    This script will:
    *   Download the FFHQ dataset (identified as `huggan/FFHQ`) from Hugging Face. This can be large and take a significant amount of time, as it initially fetches high-resolution images.
    *   Iterate through the dataset, resize each image to 128x128 pixels.
    *   Save the thumbnails as PNG files in the `data/ffhq_thumbnails_128/` directory.

    The target directory `data/ffhq_thumbnails_128/` will be created by the script if it doesn't exist.

    *Note on Hugging Face Access:* While `huggan/FFHQ` is a public dataset, if you encounter access issues or plan to use other datasets, it's good practice to log in to the Hugging Face Hub. You can do this by running `huggingface-cli login` in your terminal and following the prompts.

3.  **Update `IMAGE_DIR` in `train.py`:**
    Once the script completes, your thumbnail images will be in `project-eidolon/data/ffhq_thumbnails_128/`.
    You should then update the `IMAGE_DIR` variable in `train.py` to point to this directory:
    ```python
    # In train.py, find and modify this line:
    IMAGE_DIR = "data/ffhq_thumbnails_128" 
    ```

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
