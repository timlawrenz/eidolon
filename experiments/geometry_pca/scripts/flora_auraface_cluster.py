#!/usr/bin/env python3
"""Extract AuraFace embeddings for all approved flora images, then cluster by identity."""
import os
import sys
import time
import sqlite3
import numpy as np
import cv2
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insightface.app import FaceAnalysis

DB_PATH = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
FACES_BASE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/faces"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data", "flora_cluster_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUTPUT_DIR, "auraface_embeddings.npy")
PATHS_CACHE = os.path.join(OUTPUT_DIR, "auraface_image_paths.pkl")

if os.path.exists(CACHE_PATH):
    print("Loading cached AuraFace embeddings...")
    embeddings = np.load(CACHE_PATH)
    with open(PATHS_CACHE, 'rb') as f:
        image_paths = pickle.load(f)
    print(f"Loaded {len(embeddings)} embeddings")
else:
    # Query all approved flora images
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&nolock=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT id FROM personas WHERE name = 'flora'")
    flora_id = cur.fetchone()[0]
    cur.execute("""
        SELECT i.image_path FROM images i
        WHERE i.persona_id = ? AND i.status = 'approved'
    """, (flora_id,))
    rows = cur.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} approved images")
    
    image_paths = []
    for (img_path,) in rows:
        full_path = os.path.join(FACES_BASE, img_path.replace("faces/", ""))
        image_paths.append(full_path)
    
    # Load AuraFace
    print("Loading AuraFace model...")
    app = FaceAnalysis(
        name='auraface', 
        root='/mnt/nas-ai-models',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(512, 512))
    
    # Extract embeddings
    embeddings = []
    n_skip = 0
    n_multi = 0
    t0 = time.time()
    
    for i, img_path in enumerate(image_paths):
        if i % 50 == 0 or i < 5:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(image_paths) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(image_paths)}] {rate:.1f} img/s, ETA: {eta:.0f}s", flush=True)
        
        img = cv2.imread(img_path)
        if img is None:
            n_skip += 1
            embeddings.append(None)
            continue
        
        faces = app.get(img)
        
        if len(faces) == 0:
            n_skip += 1
            embeddings.append(None)
        elif len(faces) > 1:
            # Multiple faces detected in what should be a face crop — take largest
            largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            embeddings.append(largest.normed_embedding)
            n_multi += 1
        else:
            embeddings.append(faces[0].normed_embedding)
        
        # Checkpoint every 500 images
        if (i + 1) % 500 == 0:
            valid_indices = [j for j, e in enumerate(embeddings) if e is not None]
            X_partial = np.stack([embeddings[j] for j in valid_indices], axis=0).astype(np.float32)
            checkpoint_path = CACHE_PATH.replace('.npy', f'_ckpt{i+1}.npy')
            np.save(checkpoint_path, X_partial)
            print(f"    [checkpoint saved: {len(X_partial)} embeddings]", flush=True)
    
    elapsed = time.time() - t0
    valid = sum(1 for e in embeddings if e is not None)
    print(f"\nExtraction complete: {valid}/{len(embeddings)} valid ({n_skip} skipped, {n_multi} multi-face)")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(embeddings):.2f}s/img)")
    
    # Convert to numpy, keeping only valid
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    X = np.stack([embeddings[i] for i in valid_indices], axis=0).astype(np.float32)
    valid_paths = [image_paths[i] for i in valid_indices]
    
    # Save
    np.save(CACHE_PATH, X)
    with open(PATHS_CACHE, 'wb') as f:
        pickle.dump(valid_paths, f)
    
    embeddings = X
    image_paths = valid_paths
    print(f"Saved {len(embeddings)} embeddings to {CACHE_PATH}")

# --- Clustering ---
print(f"\n=== Identity Clustering ({len(embeddings)} embeddings) ===")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

X = embeddings

# Try k=2 through k=5
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    counts = [np.sum(labels == i) for i in range(k)]
    sil = silhouette_score(X, labels)
    
    print(f"\nk={k}: sizes={counts}, silhouette={sil:.4f}")
    
    # For each cluster, show top-2 closest to centroid
    for c in range(k):
        mask = labels == c
        cluster_embs = X[mask]
        indices = np.where(mask)[0]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        top = np.argsort(dists)[:2]
        
        print(f"  Cluster {c} (n={counts[c]}):")
        for j, idx in enumerate(top):
            global_idx = indices[idx]
            print(f"    #{j+1}: dist={dists[idx]:.5f}")
            print(f"         MEDIA:{image_paths[global_idx]}")

# Also compute pairwise cosine distance distribution for face_1 vs face_2+
# Extract face index info from paths
face_indices = []
for p in image_paths:
    basename = p.split('/')[-1]
    if '_face' in basename:
        fi = int(basename.split('_face')[1].replace('.jpg', ''))
    else:
        fi = 1
    face_indices.append(fi)
face_indices = np.array(face_indices)

# Show intra/inter face_index similarity
print("\n=== Similarity by face_index ===")
for fi in sorted(set(face_indices)):
    mask = face_indices == fi
    if mask.sum() < 2:
        continue
    # Mean pairwise cosine similarity within this group
    sims = pairwise_distances(X[mask], X[mask], metric='cosine')
    # Upper triangle only
    triu = sims[np.triu_indices(len(sims), k=1)]
    print(f"  face_{fi} (n={mask.sum()}): mean intra cosine dist = {triu.mean():.4f}")

# Save labels for k=2
k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = k2.fit_predict(X)
np.save(os.path.join(OUTPUT_DIR, "auraface_labels_k2.npy"), labels_k2)

print("\nDone! Labels saved.")
