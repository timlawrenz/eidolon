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

def scale_and_center_landmarks(avg_landmarks, out_size=(300, 300)):
    nose_tip = avg_landmarks[30]
    min_c = avg_landmarks.min(axis=0)
    max_c = avg_landmarks.max(axis=0)
    size_c = max_c - min_c
    scale = (min(out_size) * 0.8) / max(size_c)

    scaled_avg = np.zeros_like(avg_landmarks)
    scaled_avg[:, 0] = (out_size[0] / 2.0) + (avg_landmarks[:, 0] - nose_tip[0]) * scale
    scaled_avg[:, 1] = (out_size[1] / 2.0) + (avg_landmarks[:, 1] - nose_tip[1]) * scale
    return scaled_avg

def generate_pixel_average(image_paths, landmarks_list, avg_landmarks, out_size=(300, 300)):
    scaled_avg = scale_and_center_landmarks(avg_landmarks, out_size)

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
        
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            continue
            
        src_px = np.zeros_like(src_points)
        src_px[:, 0] = (src_points[:, 0] / 2.0 + 0.5) * w
        src_px[:, 1] = (src_points[:, 1] / 2.0 + 0.5) * h
        
        warped_img = np.zeros((out_size[1], out_size[0], 3), np.float32)
        
        try:
            for t in tri.simplices:
                t_src = [src_px[t[0]], src_px[t[1]], src_px[t[2]]]
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
