import cv2
import numpy as np
from scipy.spatial import Delaunay

def warp_triangle(img1, img2, t1, t2):
    t1_np = np.float32(t1)
    t2_np = np.float32(t2)
    
    r1 = cv2.boundingRect(t1_np)
    r2 = cv2.boundingRect(t2_np)
    
    t1_rect = [] 
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32([t2_rect_int]), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2Rect = cv2.warpAffine(img1Rect, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    img2Rect = img2Rect * mask

    # Protect against index out of bounds if rect exceeds image
    try:
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
    except ValueError:
        pass

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

def generate_pixel_average(image_paths, landmarks_list, avg_landmarks, out_size=(300, 300)):
    # 1. Add "forehead" and "corner" points to create a proper bounding box
    # The standard 68-point model doesn't cover the forehead or the top corners.
    # We create synthetic points above the eyebrows to pull the triangulation up and out.
    
    # Calculate bounding box of the base landmarks
    min_c = avg_landmarks.min(axis=0)
    max_c = avg_landmarks.max(axis=0)
    w = max_c[0] - min_c[0]
    h = max_c[1] - min_c[1]
    
    # Create 4 synthetic corner points that pad the bounding box by 20%
    pad_w = w * 0.2
    pad_h = h * 0.2
    
    # Define corners in the original unscaled geometry space
    corners = np.array([
        [min_c[0] - pad_w, min_c[1] - pad_h], # Top left
        [max_c[0] + pad_w, min_c[1] - pad_h], # Top right
        [min_c[0] - pad_w, max_c[1] + pad_h], # Bottom left
        [max_c[0] + pad_w, max_c[1] + pad_h]  # Bottom right
    ])
    
    augmented_avg = np.vstack([avg_landmarks, corners])
    scaled_avg = scale_and_center_landmarks(augmented_avg, out_size)

    output = np.zeros((out_size[1], out_size[0], 3), np.float32)
    try:
        tri = Delaunay(scaled_avg)
    except Exception as e:
        print(f"    [!] Delaunay triangulation failed: {e}")
        return None
    
    count = 0
    for img_path, src_points in zip(image_paths, landmarks_list):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img = np.float32(img) / 255.0
        
        h_img, w_img = img.shape[:2]
        if w_img <= 0 or h_img <= 0:
            continue
            
        src_px = np.zeros_like(src_points)
        src_px[:, 0] = (src_points[:, 0] / 2.0 + 0.5) * w_img
        src_px[:, 1] = (src_points[:, 1] / 2.0 + 0.5) * h_img
        
        # Calculate bounding box of the source landmarks
        src_min = src_px.min(axis=0)
        src_max = src_px.max(axis=0)
        src_w = src_max[0] - src_min[0]
        src_h = src_max[1] - src_min[1]
        
        src_pad_w = src_w * 0.2
        src_pad_h = src_h * 0.2
        
        src_corners = np.array([
            [src_min[0] - src_pad_w, src_min[1] - src_pad_h],
            [src_max[0] + src_pad_w, src_min[1] - src_pad_h],
            [src_min[0] - src_pad_w, src_max[1] + src_pad_h],
            [src_max[0] + src_pad_w, src_max[1] + src_pad_h]
        ])
        
        augmented_src = np.vstack([src_px, src_corners])
        
        warped_img = np.zeros((out_size[1], out_size[0], 3), np.float32)
        
        try:
            for t in tri.simplices:
                t_src = [augmented_src[t[0]], augmented_src[t[1]], augmented_src[t[2]]]
                t_dst = [scaled_avg[t[0]], scaled_avg[t[1]], scaled_avg[t[2]]]
                warp_triangle(img, warped_img, t_src, t_dst)
                
            output += warped_img
            count += 1
        except cv2.error as e:
            # Catch OpenCV cv2.warpAffine assertion errors silently and just skip this one image
            pass
        
    if count > 0:
        output = output / count
        
    return (output * 255).astype(np.uint8)
