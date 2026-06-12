#!/usr/bin/env python3
import os, sqlite3, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.verification import verification_auc
from geometry_pca.zg_inference import encode_zg
from geometry_pca.normal_encoder import derive_variant

FACE = "/mnt/nas-ai-models/training-data/eidolon/geometry_pca_data/hegre_faces_stratum"
CONF = 0.5
SEEDS = [0, 1, 2]

# Load PCA encoders
enc_zg = dict(np.load("output/encoder_production.npz"))
enc_za = dict(np.load("output/encoder_za_xy.npz"))

db = sqlite3.connect("file:data/review.db?mode=ro", uri=True)
c = db.cursor()
c.execute("""
    SELECT i.enriched_dir, i.persona_id
    FROM images i
    WHERE i.status='approved'
      AND i.persona_id NOT IN (SELECT persona_id FROM images WHERE status='tainted:contamination')
    ORDER BY i.persona_id, i.id
""")
rows = c.fetchall()
db.close()

X_dino, Y_zg, Y_za, groups = [], [], [], []
for ed_rel, pid in rows:
    leaf = ed_rel.split("hegre_enriched/", 1)[1]
    ed = os.path.join(FACE, leaf)
    try:
        dino = np.load(os.path.join(ed, "dinov3_cls.npy")).astype(np.float32)
        pose = np.load(os.path.join(ed, "pose.npy")).astype(np.float32)
        if pose[23:91, 2].mean() < CONF: continue
        normal = np.load(os.path.join(ed, "normal.npy")).astype(np.float32)
    except Exception:
        continue
    
    # Encode zg (includes pose extraction & whitening)
    face_2d = pose[23:91, :2]
    zg = encode_zg(face_2d, enc_zg)
    if zg is None: continue
    
    # Encode za
    from geometry_pca.depth_encoder import face_bbox_px
    from geometry_pca.normal_encoder import resample_masked_3ch, head_rotation
    from geometry_pca.canonical_face import canonical_template

    CANONICAL_TPL = canonical_template()

    # Create foreground mask (Sapiens body classes > 0)
    seg = np.load(os.path.join(ed, "seg.npy"))
    fgmask = (seg > 0).astype(np.float32)
    mag = np.linalg.norm(normal, axis=-1)
    fgmask *= (mag > 0.1).astype(np.float32)
    if fgmask.sum() < 50: continue
    
    h, w = normal.shape[:2]
    face_r3 = pose[23:91]
    x0, y0, x1, y1 = face_bbox_px(face_r3, h, w)
    
    normal_bg = normal.copy()
    normal_bg[fgmask < 0.5] = 0.0
    grid = resample_masked_3ch(normal_bg, x0, y0, x1, y1, out_res=64)
    R = head_rotation(face_r3[:, :2], CANONICAL_TPL)
    
    v = derive_variant(grid, R, "rot_xy").reshape(1, -1)
    za = (v - enc_za["pca_mean"]) @ enc_za["components"].T
    za = (za - enc_za["whiten_mu"]) / enc_za["whiten_sigma"]
    
    X_dino.append(dino)
    Y_zg.append(zg.ravel())
    Y_za.append(za.ravel())
    groups.append(pid)

X_dino = np.stack(X_dino)
Y_zg = np.stack(Y_zg)
Y_za = np.stack(Y_za)
groups = np.array(groups)

print(f"Loaded {len(X_dino)} images, {len(np.unique(groups))} identities.")

# Group K-Fold CV
gkf = GroupKFold(n_splits=5)
Y_pred_g = np.zeros_like(Y_zg)
Y_pred_a = np.zeros_like(Y_za)

for train_idx, test_idx in gkf.split(X_dino, groups=groups):
    m_g = Ridge(alpha=100.0).fit(X_dino[train_idx], Y_zg[train_idx])
    Y_pred_g[test_idx] = m_g.predict(X_dino[test_idx])
    
    m_a = Ridge(alpha=100.0).fit(X_dino[train_idx], Y_za[train_idx])
    Y_pred_a[test_idx] = m_a.predict(X_dino[test_idx])

# Verification AUC
auc_g = float(np.mean([verification_auc(Y_pred_g, groups, seed=s)[0] for s in SEEDS]))
auc_a = float(np.mean([verification_auc(Y_pred_a, groups, seed=s)[0] for s in SEEDS]))

print("\n[HEGRE-FIT BRIDGE CONTROL (5-fold CV)]")
print(f"  Ŷ_g (hegre-fit) AUC : {auc_g:.4f}")
print(f"  Ŷ_a (hegre-fit) AUC : {auc_a:.4f}")

# Re-run random projection null for this exact set
proj_aucs = []
for ps in range(5):
    P = np.random.default_rng(1000 + ps).normal(size=(1024, 50)).astype(np.float32) / np.sqrt(1024)
    Z = X_dino @ P
    proj_aucs.append(float(np.mean([verification_auc(Z, groups, seed=s)[0] for s in SEEDS])))
null_mean = float(np.mean(proj_aucs))
print(f"\n  AUC(random_proj null) : {null_mean:.4f} ± {np.std(proj_aucs):.4f}")
