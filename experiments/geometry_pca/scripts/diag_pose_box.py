import numpy as np

pose = np.load("experiments/geometry_pca/data/hegre_enriched/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px/pose.npy", allow_pickle=True)
# Pose shape is usually (N, 133, 3) where 3 is (x, y, confidence)
if pose.ndim == 2 and pose.shape[0] == 133:
    pose = np.expand_dims(pose, 0)

# DWPose face keypoints are usually 23 to 90 (68 points)
# Let's just use all keypoints above confidence 0.3 for the face
# Actually, the standard whole-body is 133.
# Let's find the head by looking at points 0 (nose), 1 (left eye), 2 (right eye), 3 (left ear), 4 (right ear)
for i, p in enumerate(pose):
    face_kpts = p[24:92] # Standard 68 face points
    # Filter by confidence
    valid = face_kpts[face_kpts[:, 2] > 0.3]
    if len(valid) > 10:
        x_min, y_min = valid[:, 0].min(), valid[:, 1].min()
        x_max, y_max = valid[:, 0].max(), valid[:, 1].max()
        print(f"Person {i} Face Box (normalized):", x_min, y_min, x_max, y_max)
        
        # Absolute coords
        w, h = 10246, 13662
        print(f"Person {i} Face Box (absolute):", x_min*w, y_min*h, x_max*w, y_max*h)

