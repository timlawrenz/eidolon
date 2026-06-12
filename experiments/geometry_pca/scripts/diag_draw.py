import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
img_path = "/mnt/nas-ai-models/training-data/loras/hegre-14000px/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px.jpg"

img = Image.open(img_path).convert('RGB')
MAX_DIM = 2048
scale = min(MAX_DIM / img.width, MAX_DIM / img.height)
new_size = (int(img.width * scale), int(img.height * scale))
img_small = img.resize(new_size, Image.Resampling.BILINEAR)

boxes, probs = mtcnn.detect(img_small)
if boxes is not None:
    box = boxes[0]
    box_large = [b / scale for b in box]
    
    # Draw on a downscaled version so we can view it
    draw_img = img_small.copy()
    draw = ImageDraw.Draw(draw_img)
    draw.rectangle(box.tolist(), outline="red", width=5)
    draw_img.save("experiments/geometry_pca/data/hegre_faces/diag_draw.jpg")
    print("Saved to diag_draw.jpg")

