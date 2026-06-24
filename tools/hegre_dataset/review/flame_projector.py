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

def compute_uv_coordinates(vertices: np.ndarray, landmarks: np.ndarray, out_size: tuple[int, int] = (300, 300), target_iod_ratio=0.2) -> np.ndarray:
    """
    Map 3D vertices to 2D UV coordinates matching the Pixel Average projection.
    
    Args:
        vertices: (V, 3) float array of 3D vertices (right-handed, +Y up, +X right)
        landmarks: (68, 3) float array of 3D landmarks
        out_size: The resolution of the texture image (width, height)
        target_iod_ratio: The target inter-ocular distance as a ratio of the image width
        
    Returns:
        uvs: (V, 2) float array of UV coordinates normalized to [0, 1].
             Origin (0,0) is top-left of the texture.
    """
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
    We generate a synthetic fallback using matplotlib 3D scatter since the server lacks EGL/OSMesa OpenGL drivers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import io
    from PIL import Image
    import imageio

    # Recenter mesh to origin
    centroid = mesh.vertices.mean(axis=0)
    v = mesh.vertices - centroid
    
    # We will just plot a subset of vertices for speed and styling
    # Plotting 5000 points is fine for a static frame, but for a GIF we might want to subsample
    sub_v = v[::2] 
    
    # Extract the colors from the texture if visual is set
    colors = None
    if hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv
        img = np.array(mesh.visual.material.image)
        if img is not None and len(img.shape) == 3:
            h, w = img.shape[:2]
            u_px = np.clip((uvs[::2, 0] * w).astype(int), 0, w-1)
            v_px = np.clip((uvs[::2, 1] * h).astype(int), 0, h-1)
            colors = img[v_px, u_px] / 255.0

    if colors is None:
        colors = np.ones((len(sub_v), 3)) * 0.7
        
    frames = []
    angles = np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * np.radians(30)
    
    fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100, facecolor='#18181B') # Zinc 950
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#18181B')
    
    # Remove axes
    ax.set_axis_off()
    
    # Set fixed limits to prevent bounding box bouncing
    r = 0.08
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    
    # View angle (elevation, azimuth)
    # X=right, Y=up, Z=forward in FLAME
    # Matplotlib: Z is up. 
    # So we need to swap Y and Z for plotting
    plot_x = sub_v[:, 0]
    plot_y = sub_v[:, 2] # Forward
    plot_z = sub_v[:, 1] # Up

    scatter = ax.scatter(plot_x, plot_y, plot_z, c=colors, s=1, depthshade=True, alpha=0.8)
    
    # Set the initial view (straight on)
    ax.view_init(elev=0, azim=-90)
    
    for angle in angles:
        # We rotate by changing the azimuth
        azimuth = -90 + np.degrees(angle)
        ax.view_init(elev=0, azim=azimuth)
        
        # Render frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#18181B', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.array(Image.open(buf).convert("RGB"))
        frames.append(img_arr)
        
    plt.close(fig)
    
    imageio.mimsave(str(output_path), frames, fps=15, loop=0)

