import numpy as np
pose = np.load("experiments/geometry_pca/data/hegre_enriched/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px/pose.npy", allow_pickle=True)
if pose.ndim == 2: pose = np.expand_dims(pose, 0)
print("Nose:", pose[0][0])
print("LEye:", pose[0][1])
print("REye:", pose[0][2])
print("LEar:", pose[0][3])
print("REar:", pose[0][4])
