import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
img_path = "/mnt/nas-ai-models/training-data/loras/hegre-14000px/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px.jpg"

img = Image.open(img_path)
img = ImageOps.exif_transpose(img).convert('RGB')

boxes, probs = mtcnn.detect(img)
box = boxes[0]
x1, y1, x2, y2 = box
w, h = x2 - x1, y2 - y1
cx, cy = x1 + w/2, y1 + h/2

side = max(w, h) * 1.5

nx1 = max(0, cx - side/2)
ny1 = max(0, cy - side/2)
nx2 = min(img.width, cx + side/2)
ny2 = min(img.height, cy + side/2)

final_side = min(nx2 - nx1, ny2 - ny1)
fx1 = cx - final_side/2
fy1 = cy - final_side/2
fx2 = cx + final_side/2
fy2 = cy + final_side/2

sq_box = (max(0, int(fx1)), max(0, int(fy1)), int(fx2), int(fy2))
crop = img.crop(sq_box)
crop_resized = crop.resize((1024, 1024), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
out_path = "experiments/geometry_pca/data/hegre_faces/diag_crop.jpg"
crop_resized.save(out_path)
print("Crop saved to:", out_path)
print("Crop size:", crop_resized.size)

img_thumb = img.resize((512, int(512 * img.height / img.width)))
img_thumb.save("experiments/geometry_pca/data/hegre_faces/diag_thumb.jpg")
print("Thumb saved to: experiments/geometry_pca/data/hegre_faces/diag_thumb.jpg")
