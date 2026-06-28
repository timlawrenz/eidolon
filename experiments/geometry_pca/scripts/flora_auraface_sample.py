#!/usr/bin/env python3
"""Stratified sample of flora images + AuraFace identity clustering."""
import os
import sys
import time
import sqlite3
import numpy as np
import cv2
import pickle
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insightface.app import FaceAnalysis

DB_PATH = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
FACES_BASE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/faces"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data", "flora_cluster_analysis")

random.seed(42)

# --- Stratified sampling ---
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&nolock=1", uri=True)
cur = conn.cursor()
cur.execute("SELECT id FROM personas WHERE name='flora'")
flora_id = cur.fetchone()[0]

# Get all approved images with shoot info
cur.execute("""
    SELECT i.image_path, s.slug, i.face_index
    FROM images i
    JOIN sets s ON i.set_id = s.id
    WHERE i.persona_id = ? AND i.status = 'approved'
""", (flora_id,))
rows = cur.fetchall()
conn.close()

# Categorize
solo_f1 = []
with_alex = []
with_mike = []
with_zaika = []
with_thea = []

for img_path, slug, face_idx in rows:
    full = os.path.join(FACES_BASE, img_path.replace("faces/", ""))
    entry = (full, slug, face_idx)
    if 'alex' in slug:
        with_alex.append(entry)
    elif 'mike' in slug:
        with_mike.append(entry)
    elif 'zaika' in slug:
        with_zaika.append(entry)
    elif 'thea' in slug:
        with_thea.append(entry)
    else:
        if face_idx == 1:  # solo shoots: only take face_1
            solo_f1.append(entry)

print(f"Pool sizes: solo_f1={len(solo_f1)}, alex={len(with_alex)}, mike={len(with_mike)}, "
      f"zaika={len(with_zaika)}, thea={len(with_thea)}")

# Sample
sample = []
sample += random.sample(solo_f1, min(200, len(solo_f1)))
sample += random.sample(with_alex, min(50, len(with_alex)))
sample += random.sample(with_mike, min(50, len(with_mike)))
sample += random.sample(with_zaika, min(50, len(with_zaika)))
sample += random.sample(with_thea, min(25, len(with_thea)))

print(f"Sampled {len(sample)} images")

# --- Extract AuraFace embeddings ---
CACHE_PATH = os.path.join(OUTPUT_DIR, "auraface_sample_embeddings.npy")
PATHS_CACHE = os.path.join(OUTPUT_DIR, "auraface_sample_paths.pkl")
META_CACHE = os.path.join(OUTPUT_DIR, "auraface_sample_meta.pkl")

if os.path.exists(CACHE_PATH):
    print("Loading cached...")
    X = np.load(CACHE_PATH)
    with open(PATHS_CACHE, 'rb') as f:
        paths = pickle.load(f)
    with open(META_CACHE, 'rb') as f:
        meta = pickle.load(f)
else:
    print("Loading AuraFace...")
    app = FaceAnalysis(
        name='auraface',
        root='/mnt/nas-ai-models',
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(512, 512))
    
    embeddings = []
    valid_paths = []
    valid_meta = []
    n_skip = 0
    
    t0 = time.time()
    for i, (img_path, slug, face_idx) in enumerate(sample):
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(sample) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(sample)}] {rate:.1f} img/s, ETA: {eta:.0f}s")
        
        img = cv2.imread(img_path)
        if img is None:
            n_skip += 1
            continue
        
        try:
            faces = app.get(img)
        except Exception as e:
            n_skip += 1
            continue
        
        if len(faces) == 0:
            n_skip += 1
            continue
        
        emb = faces[0].normed_embedding
        embeddings.append(emb)
        valid_paths.append(img_path)
        valid_meta.append((slug, face_idx))
    
    elapsed = time.time() - t0
    X = np.stack(embeddings, axis=0).astype(np.float32)
    print(f"Extracted {len(X)} embeddings in {elapsed:.1f}s ({elapsed/len(sample):.2f}s/img), skipped {n_skip}")
    
    np.save(CACHE_PATH, X)
    with open(PATHS_CACHE, 'wb') as f:
        pickle.dump(valid_paths, f)
    with open(META_CACHE, 'wb') as f:
        pickle.dump(valid_meta, f)
    paths = valid_paths
    meta = valid_meta

# --- Identity Clustering ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print(f"\n=== Identity Clustering ({len(X)} embeddings) ===")

for k in range(2, 5):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    counts = [np.sum(labels == c) for c in range(k)]
    sil = silhouette_score(X, labels)
    
    print(f"\nk={k}: sizes={counts}, silhouette={sil:.4f}")
    
    for c in range(k):
        mask = labels == c
        cluster_embs = X[mask]
        indices = np.where(mask)[0]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        top = np.argsort(dists)[:2]
        
        # Check source composition
        slug_counts = {}
        face_counts = {}
        for idx in indices:
            slug, fi = meta[idx]
            stype = slug.split('-')[0] if '-' in slug else 'solo'
            for kw in ['alex', 'mike', 'zaika', 'thea']:
                if kw in slug:
                    stype = f'with-{kw}'
                    break
            slug_counts[stype] = slug_counts.get(stype, 0) + 1
            face_counts[fi] = face_counts.get(fi, 0) + 1
        
        print(f"  Cluster {c} (n={counts[c]}):")
        print(f"    Source: {slug_counts}")
        print(f"    Face idx: {face_counts}")
        for j, idx in enumerate(top):
            global_idx = indices[idx]
            print(f"    #{j+1}: dist={dists[idx]:.5f}")
            print(f"         MEDIA:{paths[global_idx]}")
    
    # Save best k
    if k == 2:
        np.save(os.path.join(OUTPUT_DIR, "auraface_sample_labels_k2.npy"), labels)
