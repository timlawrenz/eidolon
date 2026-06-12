import numpy as np
pose = np.load("experiments/geometry_pca/data/hegre_enriched/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px/pose.npy", allow_pickle=True)
print("Shape:", pose.shape if hasattr(pose, 'shape') else "No shape")
print("Content:", pose.item() if pose.size == 1 and isinstance(pose.item(), dict) else pose)
