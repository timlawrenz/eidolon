#!/usr/bin/env python3
"""Extract AuraFace embeddings for all approved flora images. Minimal version."""
import os, sys, time, sqlite3, numpy as np, cv2, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insightface.app import FaceAnalysis

DB_PATH = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/review.db"
FACES_BASE = "/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1/faces"
OUT = "data/flora_cluster_analysis"
os.makedirs(OUT, exist_ok=True)

# Query
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&nolock=1", uri=True)
cur = conn.cursor()
cur.execute("SELECT id FROM personas WHERE name='flora'")
fid = cur.fetchone()[0]
cur.execute("SELECT image_path FROM images WHERE persona_id=? AND status='approved'", (fid,))
rows = cur.fetchall()
conn.close()

paths = [os.path.join(FACES_BASE, r[0].replace("faces/", "")) for r in rows]
N = len(paths)
print(f"Processing {N} images...")

# Load model
app = FaceAnalysis(name='auraface', root='/mnt/nas-ai-models', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

# Extract
embeddings = []
valid_paths = []
n_skip = 0
t0 = time.time()

for i, p in enumerate(paths):
    if i % 100 == 0:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (N - i) / rate if rate > 0 else 0
        print(f"  [{i}/{N}] {rate:.1f} img/s, ETA: {eta:.0f}s", flush=True)
    
    img = cv2.imread(p)
    if img is None:
        n_skip += 1; continue
    faces = app.get(img)
    if len(faces) == 0:
        n_skip += 1; continue
    emb = faces[0].normed_embedding
    embeddings.append(emb)
    valid_paths.append(p)

elapsed = time.time() - t0
X = np.stack(embeddings).astype(np.float32)
print(f"\nDone: {len(X)} valid embeddings in {elapsed:.0f}s ({elapsed/N:.2f}s/img), skipped {n_skip}")

# Save
np.save(f"{OUT}/auraface_embeddings.npy", X)
with open(f"{OUT}/auraface_image_paths.pkl", 'wb') as f:
    pickle.dump(valid_paths, f)
print(f"Saved to {OUT}/auraface_embeddings.npy")
print(f"Shape: {X.shape}")
