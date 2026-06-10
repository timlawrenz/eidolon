"""Constants for the geometry PCA encoder pipeline."""

STRATUM_ROOT = "/mnt/nas-ai-models/training-data/ffhq/stratum"
N_SAMPLES = 70000

# COCO-WholeBody facial landmarks -> 68 iBUG points
FACE_SLICE = slice(23, 91)
N_FACE_PTS = 68

# 136 after dropping the 3rd (confidence) channel
POSE_DIM = N_FACE_PTS * 2
