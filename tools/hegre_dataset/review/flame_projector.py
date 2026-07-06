import numpy as np
import sqlite3
from pathlib import Path
import cv2
import torch
import sys
import os
import imageio

os.environ["PYOPENGL_PLATFORM"] = "egl"

# pyglet.options['headless'] must be set BEFORE trimesh/pyrender/pyglet
# are imported, otherwise OpenGL will try to create an X11 window.
try:
    import pyglet
    pyglet.options['headless'] = True
except ImportError:
    pass

try:
    import pyrender
    import trimesh
except ImportError:
    pyrender = None
    trimesh = None

# Add SMIRK to path to import the encoder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "experiments", "flame_spike", "smirk"))
try:
    from src.smirk_encoder import SmirkEncoder
except ImportError:
    SmirkEncoder = None

# ── Model singleton cache ──────────────────────────────────────────
_smirk_model = None
_smirk_checkpoint_path = None
_smirk_device_str = None


def get_smirk_model(checkpoint_path: Path, device: str | None = None):
    """Return the cached SmirkEncoder, or create and cache a new one.

    The model is cached by checkpoint path and device. Subsequent calls
    with the same arguments return the identical model instance, avoiding
    redundant backbone downloads and GPU memory allocations.
    """
    global _smirk_model, _smirk_checkpoint_path, _smirk_device_str

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # If the checkpoint or device changed, create a new model
    if _smirk_model is not None and _smirk_checkpoint_path == str(checkpoint_path) and _smirk_device_str == device:
        return _smirk_model

    model = SmirkEncoder(n_shape=300).to(device)
    model.eval()

    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("smirk_encoder."):
                encoder_state_dict[k.replace("smirk_encoder.", "")] = v
        model.load_state_dict(encoder_state_dict)

    _smirk_model = model
    _smirk_checkpoint_path = str(checkpoint_path)
    _smirk_device_str = device

    return model

def compute_uv_coordinates(vertices: np.ndarray, landmarks: np.ndarray, out_size: tuple[int, int] = (300, 300), target_iod_ratio=0.2) -> np.ndarray:
    """
    Map 3D vertices to 2D UV coordinates matching the Pixel Average projection.
    
    IMPORTANT: out_size defaults to (300, 300) to match the review UI's
    THUMB_SIZE and pixel average canvas.  These three values must stay in sync.
    See ui.py module docstring for details.
    
    Args:
        vertices: (V, 3) float array of 3D vertices (right-handed, +Y up, +X right)
        landmarks: (68, 3) float array of 3D landmarks
        out_size: The resolution of the texture image (width, height)
        target_iod_ratio: The target inter-ocular distance as a ratio of the image width
        
    Returns:
        uvs: (V, 2) float array of UV coordinates normalized to [0, 1].
             Origin (0,0) is top-left of the texture.
    
    Raises:
        ValueError: if the inter-ocular distance is degenerate (<1e-6).
    """
    nose_3d = landmarks[30]
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    
    d_x = right_eye[0] - left_eye[0]
    d_y = right_eye[1] - left_eye[1]
    current_iod = np.hypot(d_x, d_y)
    
    if current_iod < 1e-6:
        raise ValueError(f"Degenerate landmarks: inter-ocular distance is {current_iod}. "
                         "This typically indicates all-zero or collapsed DWPose keypoints.")
    
    angle = np.arctan2(d_y, d_x)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    v_2d = vertices[:, :2] - nose_3d[:2]
    v_rotated = v_2d @ R.T
    
    target_uv_iod = target_iod_ratio
    scale = target_uv_iod / current_iod
    
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    uvs[:, 0] = 0.5 + v_rotated[:, 0] * scale
    uvs[:, 1] = 0.5 + v_rotated[:, 1] * scale # Do not flip Y. In OpenGL, V=1 is TOP, V=0 is BOTTOM.
    
    uvs = np.clip(uvs, 0.0, 1.0)
    return uvs

def crop_for_smirk(img: np.ndarray, face_2d: np.ndarray, target_size: int = 224) -> torch.Tensor | None:
    """Tight crop resized to 224x224 for SMIRK, matching eval_flame.py logic."""
    h, w = img.shape[:2]
    
    # Bounding box logic matching depth_encoder
    # NOTE: Stratum DWPose coordinates are normalized [-1, 1], so we must un-normalize to pixel space
    face_px = np.zeros_like(face_2d)
    face_px[:, 0] = (face_2d[:, 0] / 2.0 + 0.5) * w
    face_px[:, 1] = (face_2d[:, 1] / 2.0 + 0.5) * h
    
    min_c = face_px.min(axis=0)
    max_c = face_px.max(axis=0)
    box_w = max_c[0] - min_c[0]
    box_h = max_c[1] - min_c[1]
    
    pad = 0.2
    x0 = int(max(0, min_c[0] - box_w * pad))
    y0 = int(max(0, min_c[1] - box_h * pad))
    x1 = int(min(w, max_c[0] + box_w * pad))
    y1 = int(min(h, max_c[1] + box_h * pad))
    
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
        
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Transforms: RGB, [0, 1], ImageNet normalize
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    
    crop = np.transpose(crop, (2, 0, 1))
    return torch.from_numpy(crop).unsqueeze(0)

def extract_canonical_shape(ds: "HegreDataset", persona_name: str) -> np.ndarray:
    """
    Extracts the average 300-d FLAME shape (beta) parameter for a persona using SMIRK.
    """
    from ..dataset import HegreDataset
    rows = ds.db.execute(
        "SELECT i.image_path, p.name "
        "FROM images i JOIN personas p ON i.persona_id = p.id "
        "WHERE p.name = ? AND i.status = 'approved'",
        (persona_name,)
    ).fetchall()
    
    if not rows:
        raise ValueError(f"No approved images found for {persona_name}")

    print(f"  extract_canonical_shape: {len(rows)} approved images for '{persona_name}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = ds.root.parent.parent.parent / "experiments/flame_spike/smirk/pretrained_models/SMIRK_em1.pt"
    model = get_smirk_model(checkpoint_path, str(device))
    
    shapes = []
    skipped = 0
    
    stratum_root = ds.root / "stratum"
    base_pname = persona_name.split("_cluster_")[0]
    persona_dir = stratum_root / base_pname
    
    with torch.no_grad():
        for row in rows:
            img_path = ds.root / row["image_path"]
            img_stem = Path(row["image_path"]).stem
            
            # Use rglob to find pose.npy regardless of faces/ prefix variation
            pose_path = None
            if persona_dir.exists():
                for pth in persona_dir.rglob(f"{img_stem}/pose.npy"):
                    pose_path = pth
                    break
            
            if not img_path.exists() or pose_path is None:
                skipped += 1
                continue
                
            img = cv2.imread(str(img_path))
            pose = np.load(str(pose_path))
            
            # Keypoints are 23:91 in Stratum DWPose layout
            face_2d = pose[23:91, :2]
            
            crop_tensor = crop_for_smirk(img, face_2d)
            if crop_tensor is None:
                skipped += 1
                continue
                
            crop_tensor = crop_tensor.to(device)
            outputs = model(crop_tensor)
            
            # outputs['shape_params'] is (1, 300)
            shape_vec = outputs["shape_params"].cpu().numpy()[0]
            shapes.append(shape_vec)
            
    if not shapes:
        raise ValueError(f"Failed to extract shape vectors for {persona_name} "
                         f"({len(rows)} images found, {skipped} skipped, 0 succeeded)")
        
    # Average the beta vectors
    print(f"  extract_canonical_shape: {len(shapes)} shapes extracted, {skipped} skipped for '{persona_name}'")
    return np.mean(shapes, axis=0)

def generate_textured_mesh(avg_shape: np.ndarray, pixel_avg_path: Path) -> trimesh.Trimesh:
    """
    Generates a 3D FLAME mesh from the shape parameter and UV-maps the pixel average onto it.
    """
    if trimesh is None:
        raise ImportError("trimesh is required to generate the textured mesh")
        
    # Import the FLAME layer from SMIRK
    smirk_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "experiments", "flame_spike", "smirk")
    if smirk_path not in sys.path:
        sys.path.insert(0, smirk_path)
    # Monkey patch NumPy 2.0 removals that SMIRK expects.
    # Each alias is guarded individually so a missing alias doesn't crash the block.
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'bool_'):
        np.bool_ = bool
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
    if not hasattr(np, 'object_'):
        np.object_ = object
    if not hasattr(np, 'unicode_'):
        np.unicode_ = str
    if not hasattr(np, 'str_'):
        np.str_ = str

    from src.FLAME.FLAME import FLAME
    
    device = torch.device("cpu")
    
    flame_model_path = os.path.join(smirk_path, "assets", "FLAME2020", "generic_model.pkl")
    
    # FLAME.py hardcodes relative paths like 'assets/l_eyelid.npy', so we must chdir to SMIRK temporarily
    cwd = os.getcwd()
    os.chdir(smirk_path)
    try:
        flame_layer = FLAME(flame_model_path=flame_model_path, n_shape=300, n_exp=50).to(device)
    finally:
        os.chdir(cwd)
    
    # Zero out pose, expression, global rotation
    shape_tensor = torch.from_numpy(avg_shape).unsqueeze(0).float().to(device)
    exp_tensor = torch.zeros((1, 50), dtype=torch.float32).to(device)
    pose_params = torch.zeros((1, 3), dtype=torch.float32).to(device)
    neck_pose_params = torch.zeros((1, 3), dtype=torch.float32).to(device)
    jaw_params = torch.zeros((1, 3), dtype=torch.float32).to(device)
    eye_pose_params = torch.zeros((1, 6), dtype=torch.float32).to(device)
    eyelid_params = torch.zeros((1, 2), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        param_dict = {
            'shape_params': shape_tensor,
            'expression_params': exp_tensor,
            'pose_params': pose_params,
            'neck_pose_params': neck_pose_params,
            'jaw_params': jaw_params,
            'eye_pose_params': eye_pose_params,
            'eyelid_params': eyelid_params
        }
        outputs = flame_layer(param_dict)
        vertices = outputs['vertices']
        landmarks = outputs['landmarks_fan_3d']
        
    v = vertices[0].cpu().numpy()
    lm = landmarks[0].cpu().numpy()
    faces = flame_layer.faces_tensor.cpu().numpy()
    
    uvs = compute_uv_coordinates(v, lm, out_size=(300, 300))
    
    import pyrender
    
    # Create the texture material
    img = cv2.imread(str(pixel_avg_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
    # Pyrender handles the Trimesh visual mapping natively if we pass the visual object
    material = trimesh.visual.material.SimpleMaterial(image=img)
    visuals = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
    
    mesh = trimesh.Trimesh(vertices=v, faces=faces, visual=visuals, process=False)
    
    return mesh


def render_spin_gif(mesh: trimesh.Trimesh, output_path: Path, num_frames: int = 30, resolution: tuple[int, int] = (300, 300)):
    """
    Renders a spinning animation of the mesh into a GIF.
    Uses trimesh offscreen rendering via OSMesa.
    """
    import PIL.Image
    
    # Fix trimesh numpy 2.0 issue in power_resize
    import trimesh.visual.texture
    def patched_power_resize(image, square=False, resample=1):
        size = np.array(image.size)
        new_size = (2 ** np.round(np.log2(size))).astype(int)
        if square:
            new_size = np.array([new_size.max()] * 2)
        if (new_size == size).all():
            return image
        return image.resize((int(new_size[0]), int(new_size[1])), resample=resample)
    trimesh.visual.texture.power_resize = patched_power_resize

    # Ensure material image is PIL (cv2 reads as numpy)
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
        if isinstance(mesh.visual.material.image, np.ndarray):
            mesh.visual.material.image = PIL.Image.fromarray(mesh.visual.material.image)

    # Recenter mesh to origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid

    # Zinc 950 background
    bg_color = [24, 24, 27, 255]
    
    # Create front mesh with texture and back mesh with solid color
    front_mask = mesh.face_normals[:, 2] >= -0.1
    back_mask = ~front_mask
    
    front_faces = mesh.faces[front_mask]
    back_faces = mesh.faces[back_mask]
    
    front_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=front_faces, process=False)
    # Reassign texture to front
    import PIL.Image
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
        img = mesh.visual.material.image
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        front_mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=img)
        
    back_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=back_faces, process=False)
    back_mesh.visual.face_colors = [39, 39, 42, 255] # Zinc 800
    
    scene = trimesh.Scene()
    front_node = scene.add_geometry(front_mesh)
    back_node = scene.add_geometry(back_mesh)
    
    # Position camera
    scene.set_camera(distance=0.25)
    
    frames = []
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    
    for angle in angles:
        # Rotate around Y axis
        rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        scene.graph.update(front_node, matrix=rot)
        scene.graph.update(back_node, matrix=rot)
        
        png_data = scene.save_image(resolution=resolution, background=bg_color)
        
        # Convert PNG bytes to imageio-compatible array
        import io
        img = PIL.Image.open(io.BytesIO(png_data)).convert('RGB')
        frames.append(np.array(img))
        
    imageio.mimsave(str(output_path), frames, fps=15, loop=0)
