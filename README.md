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

### Preparing Image Dataset and Landmarks (Example: FFHQ Thumbnails via Hugging Face)

To train the encoder, you need a dataset of face images and their corresponding 2D landmarks. The script `scripts/download_ffhq_huggingface.py` handles downloading images from Hugging Face and pre-processing landmarks.

1.  **Install Dependencies:**
    Ensure your virtual environment is active and all dependencies, including `datasets` and `face_alignment`, are installed:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Data Preparation Script:**
    The script `scripts/download_ffhq_huggingface.py` will:
    *   Download the FFHQ dataset (identified as `nuwandaa/ffhq128` by default) from Hugging Face. This dataset provides images pre-resized to 128x128 pixels.
    *   Save these images as PNG files into the `data/ffhq_thumbnails_128/` directory (configurable via `IMAGE_SAVE_DIR` in the script).
    *   For each image, detect 2D facial landmarks using the `face_alignment` library.
    *   Save the detected landmarks (typically a `(68, 2)` NumPy array) as an `.npy` file in the `data/ffhq_landmarks_128/` directory (configurable via `LANDMARK_SAVE_DIR` in the script). Landmark files are named to correspond with image files (e.g., `thumb_00000.npy`).

    Run the script from the root of your `project-eidolon` directory:
    ```bash
    python scripts/download_ffhq_huggingface.py
    ```
    This process can take a significant amount of time. The script will create the necessary save directories if they don't exist.

    *Note on Hugging Face Access:* While `nuwandaa/ffhq128` is public, if you encounter access issues or use other datasets, log in via `huggingface-cli login`.

### Running the Training Script (Skeleton)

The `train.py` script trains the `EidolonEncoder`. It now uses pre-loaded images and landmarks from disk, and a multi-process `DataLoader` for efficiency.

1.  Ensure you have completed all steps in the [Setup](#setup).
2.  **Ensure you have run `scripts/download_ffhq_huggingface.py`** to prepare the image and landmark data. The output directories should be `data/ffhq_thumbnails_128/` for images and `data/ffhq_landmarks_128/` for landmarks, or as configured in `train.py`.
3.  Verify that `IMAGE_DIR` and `LANDMARK_DIR` in `train.py` point to these directories.
4.  Activate your virtual environment:
    ```bash
    python train.py
    ```
    This script will attempt to load data and start a training loop. It now includes a more complete forward pass, including calls to a placeholder `FLAME` model (from `src/model.py`) and a PyTorch3D renderer. For meaningful training that actually learns to reconstruct faces, the `FLAME` model in `src/model.py` needs to be fully implemented with the correct FLAME deformation logic. 
    
    During training, the script will:
    *   Print basic progress information (loss, batch size) to the console.
    *   Periodically save visual validation samples (ground truth vs. prediction with landmarks) to the `outputs/validation_images/` directory.
    *   Log training metrics (losses, learning rate) to TensorBoard in the `runs/project_eidolon_experiment1/` directory. You can view these logs by running `tensorboard --logdir runs` from the project root and opening the provided URL in your browser.
    *   Log validation images to TensorBoard under the "Images" tab.
    
    Upon completion, the trained encoder model weights will be saved to `eidolon_encoder_final.pth` in the project root directory.

### Multi-Stage Training

To help stabilize training and guide the model towards a good solution, `train.py` supports a multi-stage training approach. This allows you to define different sets of loss weights and numbers of epochs for distinct phases of training.

**Rationale:**
-   **Stage 1 (Stabilization):** In the initial stage, the goal is to get the model to produce roughly correct outputs. This typically involves:
    -   Strong landmark loss to ensure basic 2D alignment.
    -   Strong regularization for global pose (towards identity), jaw/neck/eye poses (towards neutral), shape (towards average), and translation (towards center). This prevents extreme, non-face-like predictions.
-   **Stage 2+ (Finetuning):** Once the model is stable, subsequent stages can gradually relax the regularization terms. This allows the model to learn more detailed variations in shape, pose, and potentially expression (if enabled). The landmark loss weight might also be reduced as the model gets better at pixel-level reconstruction.

**Configuration:**
The multi-stage training is configured in `train.py` via the `TRAINING_STAGES` list. Each element in this list is a dictionary defining a stage:
```python
TRAINING_STAGES = [
    {
        'name': 'Stage1_StabilizePose', # A descriptive name for the stage
        'epochs': 20,                  # Number of epochs for this specific stage
        'loss_weights': {              # Dictionary of loss weights for this stage
            'pixel': 1.0,
            'landmark': 1.0,
            'reg_shape': 0.5,
            'reg_transl': 0.5,
            'reg_global_pose': 1.0,
            'reg_jaw_pose': 1.0,
            'reg_neck_pose': 0.5,
            'reg_eye_pose': 0.5,
        }
    },
    {
        'name': 'Stage2_FinetuneDetails',
        'epochs': 30,
        'loss_weights': {
            'pixel': 1.0,
            'landmark': 1e-1,
            # ... other relaxed weights ...
        }
    }
    # Add more stages as needed
]
```
The script will iterate through these stages, applying the specified number of epochs and loss weights for each. The total number of epochs (`NUM_EPOCHS`) is automatically calculated as the sum of epochs from all defined stages. TensorBoard logs will also be organized to reflect these stages for easier analysis.

You can customize the number of stages, the epochs per stage, and the specific loss weights to suit your training needs and observations.

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
