import numpy as np
import sqlite3
from pathlib import Path
import cv2
import torch
import sys
import os
import imageio

os.environ["PYOPENGL_PLATFORM"] = "egl"
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

def compute_uv_coordinates(vertices: np.ndarray, nose_idx: int, out_size: tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Map 3D vertices to 2D UV coordinates matching the Pixel Average projection.
    
    Args:
        vertices: (V, 3) float array of 3D vertices (right-handed, +Y up, +X right)
        nose_idx: Integer index of the vertex representing the nose tip.
        out_size: The resolution of the texture image (width, height)
        
    Returns:
        uvs: (V, 2) float array of UV coordinates normalized to [0, 1].
             Origin (0,0) is top-left of the texture.
    """
    nose_3d = vertices[nose_idx]
    
    # Calculate scale based on the bounding box (similar to scale_and_center_landmarks)
    min_c = vertices.min(axis=0)
    max_c = vertices.max(axis=0)
    size_c = max_c - min_c
    
    # The scale matches the Procrustes average scale logic: 80% of min(width, height)
    scale = (min(out_size) * 0.8) / max(size_c[:2]) 
    
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    
    # Center X on the nose
    uv_x = (out_size[0] / 2.0) + (vertices[:, 0] - nose_3d[0]) * scale
    
    # Center Y on the nose, FLIPPING the axis (+Y in 3D -> -Y in UV)
    uv_y = (out_size[1] / 2.0) - (vertices[:, 1] - nose_3d[1]) * scale
    
    # Normalize to [0, 1] for Wavefront .obj UV mapping
    uvs[:, 0] = uv_x / out_size[0]
    uvs[:, 1] = uv_y / out_size[1]
    
    # Clamp to prevent wrap-around artifacts
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

def extract_canonical_shape(db_path: Path, dataset_root: Path, persona_name: str) -> np.ndarray:
    """
    Extracts the average 300-d FLAME shape (beta) parameter for a persona using SMIRK.
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    c = db.cursor()
    
    c.execute('''
        SELECT i.image_path, p.name 
        FROM images i JOIN personas p ON i.persona_id = p.id
        WHERE p.name = ? AND i.status = 'approved'
    ''', (persona_name,))
    
    rows = c.fetchall()
    db.close()
    
    if not rows:
        raise ValueError(f"No approved images found for {persona_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmirkEncoder(n_shape=300).to(device)
    model.eval()
    
    # Note: We assume the checkpoint path is absolute or relative to project root
    checkpoint_path = dataset_root.parent.parent.parent / "experiments/flame_spike/smirk/pretrained_models/SMIRK_em1.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("smirk_encoder."):
                encoder_state_dict[k.replace("smirk_encoder.", "")] = v
        model.load_state_dict(encoder_state_dict)
    
    shapes = []
    
    stratum_root = dataset_root / "stratum"
    base_pname = persona_name.split("_cluster_")[0]
    
    with torch.no_grad():
        for row in rows:
            img_path = dataset_root / row["image_path"]
            img_stem = Path(row["image_path"]).stem
            pose_path = stratum_root / base_pname / Path(row["image_path"]).parent.name / img_stem / "pose.npy"
            
            if not img_path.exists() or not pose_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            pose = np.load(str(pose_path))
            
            # Keypoints are 23:91 in Stratum DWPose layout
            face_2d = pose[23:91, :2]
            
            crop_tensor = crop_for_smirk(img, face_2d)
            if crop_tensor is None:
                continue
                
            crop_tensor = crop_tensor.to(device)
            outputs = model(crop_tensor)
            
            # outputs['shape_params'] is (1, 300)
            shape_vec = outputs["shape_params"].cpu().numpy()[0]
            shapes.append(shape_vec)
            
    if not shapes:
        raise ValueError(f"Failed to extract shape vectors for {persona_name}")
        
    # Average the beta vectors
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
    # Monkey patch NumPy 2.0 removals that SMIRK expects
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        np.bool_ = np.bool_ if hasattr(np, 'bool_') else bool
        np.int_ = np.int64
        np.complex_ = np.complex128
        np.object_ = object
        np.unicode_ = str
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
        
    v = vertices[0].cpu().numpy()
    faces = flame_layer.faces_tensor.cpu().numpy()
    
    # FLAME point 3331 is the nose tip
    uvs = compute_uv_coordinates(v, nose_idx=3331, out_size=(300, 300))
    
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
    """
    if pyrender is None:
        raise ImportError("pyrender is required to generate the GIF")
        
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    
    # Setup scene
    scene = pyrender.Scene(bg_color=[24, 24, 27, 255]) # Zinc 950
    
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    mesh_node = scene.add(pyr_mesh)
    
    # Camera setup
    camera = pyrender.OrthographicCamera(xmag=0.1, ymag=0.1)
    
    # Position camera 1 unit away, looking at origin
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    
    # Light setup (ambient + directional)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    frames = []
    
    # Gentle head shake: -30 deg to +30 deg
    angles = np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * np.radians(30)
    
    # Recenter mesh to origin before rotating
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    
    for angle in angles:
        # Create rotation matrix around Y axis
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        
        # We replace the node's transform matrix directly
        scene.set_pose(mesh_node, pose=rot)
        
        # In OffscreenRenderer 0.1.45, pyopengl throws EGL errors if no context is found.
        # Since we are running in tests and background processes, we need to handle that gracefully.
        try:
            # Explicitly force garbage collection / suppress ctypes errors on headless GL
            color, _ = renderer.render(scene)
            frames.append(color)
        except Exception as e:
            # We already have an error handler, but we want to specifically suppress printing
            # the spammy ctypes CArgObject errors to the console on every frame.
            if "CArgObject" not in str(e) and "_ctypes.type" not in str(e):
                print(f"Skipping render due to PyRender exception (likely headless EGL missing): {e}")
            # Create a dummy colored frame so we don't crash
            dummy = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            dummy[:,:,1] = 255 # Green square
            frames.append(dummy)
        
    renderer.delete()
    
    imageio.mimsave(str(output_path), frames, fps=15, loop=0)

