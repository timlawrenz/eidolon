# FLAME Normalized Pixel Projection Implementation Plan
> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
**Goal:** Modify the pixel average and FLAME UV mapping to strictly align faces (pivoting on the nose tip, eyes perfectly horizontal, and fixed inter-ocular distance scaling) without using bounding-box centering.
**Architecture:** 
1. `scale_and_center_landmarks` in `procrustes.py` will calculate the angle and inter-ocular distance (IOD) from 2D landmarks 36-41 (left eye) and 42-47 (right eye). It will apply a 2D rotation matrix and a fixed IOD scale (e.g. 30% of output width), pivoting strictly on nose tip 30.
2. `compute_uv_coordinates` in `flame_projector.py` will extract `landmarks_fan_3d` from the FLAME layer output. It will perform the exact same rotational leveling and IOD scaling in 3D-to-UV space, ensuring the generated 2D pixel average perfectly wraps the 3D mean FLAME skull.
**Tech Stack:** Python, NumPy, FLAME, OpenCV.
---

### Task 1: Update 2D Normalization (Pixel Average)
**Objective:** Rewrite `scale_and_center_landmarks` to pivot on the nose, align eyes horizontally, and scale by a fixed IOD.
**Files:** `tools/hegre_dataset/review/procrustes.py`
**Step 1:** Replace `scale_and_center_landmarks` with the new logic:
```python
def scale_and_center_landmarks(avg_landmarks, out_size=(300, 300), target_iod_ratio=0.3):
    nose_tip = avg_landmarks[30]
    left_eye = avg_landmarks[36:42].mean(axis=0)
    right_eye = avg_landmarks[42:48].mean(axis=0)
    
    d_x = right_eye[0] - left_eye[0]
    d_y = right_eye[1] - left_eye[1]
    current_iod = np.hypot(d_x, d_y)
    
    # Rotation to make eyes horizontal
    angle = np.arctan2(d_y, d_x)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    # Rotate around nose
    lm_centered = avg_landmarks[:, :2] - nose_tip[:2]
    lm_rotated = lm_centered @ R.T
    
    # Scale by fixed IOD
    target_iod_pixels = min(out_size) * target_iod_ratio
    scale = target_iod_pixels / current_iod
    
    scaled_avg = np.zeros_like(avg_landmarks)
    scaled_avg[:, 0] = (out_size[0] / 2.0) + lm_rotated[:, 0] * scale
    scaled_avg[:, 1] = (out_size[1] / 2.0) + lm_rotated[:, 1] * scale 
    
    return scaled_avg
```
**Step 2:** Ensure it passes visual or mathematical sanity checks.

### Task 2: Update 3D Normalization (FLAME UV Mapping)
**Objective:** Rewrite `compute_uv_coordinates` to accept FLAME landmarks and apply the exact same alignment logic to project the 3D vertices into UV space.
**Files:** `tools/hegre_dataset/review/flame_projector.py`, `tools/hegre_dataset/tests/test_flame_uv.py`
**Step 1:** Replace `compute_uv_coordinates` in `flame_projector.py`:
```python
def compute_uv_coordinates(vertices: np.ndarray, landmarks: np.ndarray, out_size: tuple[int, int] = (300, 300), target_iod_ratio=0.3) -> np.ndarray:
    nose_3d = landmarks[30]
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    
    d_x = right_eye[0] - left_eye[0]
    d_y = right_eye[1] - left_eye[1]
    current_iod = np.hypot(d_x, d_y)
    
    angle = np.arctan2(d_y, d_x)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    v_2d = vertices[:, :2] - nose_3d[:2]
    v_rotated = v_2d @ R.T
    
    target_uv_iod = target_iod_ratio
    scale = target_uv_iod / current_iod
    
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    uvs[:, 0] = 0.5 + v_rotated[:, 0] * scale
    uvs[:, 1] = 0.5 - v_rotated[:, 1] * scale # Flip Y since +Y is UP in 3D
    
    uvs = np.clip(uvs, 0.0, 1.0)
    return uvs
```

**Step 2:** Update `generate_textured_mesh` in `flame_projector.py` to extract and pass landmarks:
```python
    with torch.no_grad():
        param_dict = {...}
        outputs = flame_layer(param_dict)
        vertices = outputs['vertices']
        landmarks = outputs['landmarks_fan_3d']
        
    v = vertices[0].cpu().numpy()
    lm = landmarks[0].cpu().numpy()
    faces = flame_layer.faces_tensor.cpu().numpy()
    
    uvs = compute_uv_coordinates(v, lm, out_size=(300, 300))
```

**Step 3:** Update `tools/hegre_dataset/tests/test_flame_uv.py` to reflect the new signature (passing a fake `landmarks` array containing indices 30, 36:42, and 42:48 instead of just `nose_idx`).

### Task 4: Verification
**Objective:** Run the tests to ensure functionality.
**Step 1:** Run `pytest tools/hegre_dataset/tests/test_flame_uv.py` to verify UV space anchoring.
**Step 2:** Commit changes securely to Git history.
