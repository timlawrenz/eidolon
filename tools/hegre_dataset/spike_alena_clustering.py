import sys
import os
import json
import numpy as np
from pathlib import Path

# Fix the import path so `geometry_pca` resolves properly.
# The internal geometry_pca imports assume experiments/geometry_pca/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../experiments/geometry_pca")))

from geometry_pca.zg_inference import encode_zg
from geometry_pca.fit import load_encoder

def main():
    ds_dir = Path("data/hegre_datasets/hegre-faces/v1")
    faces_dir = ds_dir / "faces"
    stratum_dir = ds_dir / "stratum"
    
    # Check if alena poses exist
    alena_dir = stratum_dir / "alena"
    if not alena_dir.exists():
        print(f"Directory {alena_dir} not found. Searching elsewhere...")
        # Maybe we need to load from the DB first to find paths.
        
    print(f"Loading alena poses from {stratum_dir}...")
    
    pose_files = list(stratum_dir.rglob("*/alena*/**/pose.npy")) + list(stratum_dir.rglob("alena/**/pose.npy"))
    if not pose_files:
        print("No pose files found for 'alena'. Please verify stratum has processed her.")
        return
        
    print(f"Found {len(pose_files)} pose files for alena.")

    try:
        production_encoder = load_encoder("experiments/geometry_pca/output/encoder_production.npz")
    except FileNotFoundError:
        print("Error: zg_encoder.pkl not found. Please ensure geometry_pca encoders are trained.")
        return

    vectors = []
    paths = []
    
    for pf in pose_files:
        try:
            pose = np.load(pf).astype(np.float32)
            # DWPose face keypoints are usually 23:91 (68 points). Let's let encode_zg handle it or do it here if needed.
            # In loader.py, load_face_keypoints grabs 23:91. 
            face_2d = pose[23:91, :2] # Just X,Y
            zg = encode_zg(face_2d, production_encoder)
            vectors.append(zg)
            paths.append(str(pf))
        except Exception as e:
            print(f"Error processing {pf}: {e}")
            
    if not vectors:
        print("No valid vectors generated.")
        return
        
    vectors = np.array(vectors)
    print(f"Generated {vectors.shape} matrix of zg vectors for alena.")
    
    # Calculate centroid and distances
    centroid = np.mean(vectors, axis=0)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    
    # Show potential outliers
    threshold = np.mean(distances) + 2 * np.std(distances)
    outliers = [(paths[i], distances[i]) for i in range(len(distances)) if distances[i] > threshold]
    
    # Display stats
    # Filter out exact duplicates (path overlap from recursive glob)
    unique_outliers = {}
    for p, d in outliers:
        if p not in unique_outliers:
            unique_outliers[p] = d
            
    print(f"Mean distance to centroid: {np.mean(distances):.4f}")
    print(f"Std dev of distances: {np.std(distances):.4f}")
    
    # Find the image closest to the centroid
    closest_idx = np.argmin(distances)
    closest_path = paths[closest_idx]
    
    print(f"\nClosest image to centroid (Distance: {distances[closest_idx]:.4f}):")
    print(f"  - {closest_path}")
    
    print(f"\nFound {len(unique_outliers)} potential outliers (> 2 std devs from mean):")
    sorted_outliers = sorted(unique_outliers.items(), key=lambda x: x[1], reverse=True)
    for p, d in sorted_outliers[:10]:
        print(f"  - Distance: {d:.4f} | Path: {p}")
        
    # Generate the collage
    from PIL import Image, ImageDraw, ImageFont
    
    # We need to map the pose.npy paths back to the face JPEGs
    # e.g. data/.../stratum/alena/alena-alone/alena-alone-01_face1/pose.npy 
    #   -> data/.../faces/alena/alena-alone/alena-alone-01_face1.jpg
    def get_face_img_path(pose_path_str):
        p = Path(pose_path_str)
        # p.parent.name is the face dir (e.g. alena-alone-01_face1)
        # p.parent.parent.name is the set (e.g. alena-alone)
        # p.parent.parent.parent.name is the identity (e.g. alena)
        face_name = p.parent.name + ".jpg"
        set_name = p.parent.parent.name
        identity_name = p.parent.parent.parent.name
        return ds_dir / "faces" / identity_name / set_name / face_name

    images_to_show = [(closest_path, "CENTROID")] + [(p, f"OUTLIER ({d:.1f})") for p, d in sorted_outliers[:5]]
    
    thumb_size = 256
    columns = 3
    rows = 2
    
    collage = Image.new('RGB', (columns * thumb_size, rows * thumb_size), (30, 30, 30))
    draw = ImageDraw.Draw(collage)
    
    for i, (pose_path, label) in enumerate(images_to_show):
        col = i % columns
        row = i // columns
        
        face_jpg = get_face_img_path(pose_path)
        if face_jpg.exists():
            img = Image.open(face_jpg).resize((thumb_size, thumb_size))
            collage.paste(img, (col * thumb_size, row * thumb_size))
        
        # Draw label background and text
        x, y = col * thumb_size, row * thumb_size
        draw.rectangle([x, y, x + 150, y + 25], fill=(0, 0, 0, 180))
        draw.text((x + 5, y + 5), label, fill=(255, 255, 255))
        
    collage_path = ds_dir / "alena_clustering_spike.jpg"
    collage.save(collage_path)
    print(f"\nCollage saved to {collage_path.resolve()}")

if __name__ == "__main__":
    main()
