import torch
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
img_path = "/mnt/nas-ai-models/training-data/loras/hegre-14000px/6850_anna-l-hegre-model/anna-l-hegre-model-01-14000px.jpg"

img = Image.open(img_path).convert('RGB')
print("Original size:", img.size)

# Downscale for detection
MAX_DIM = 2048
scale = min(MAX_DIM / img.width, MAX_DIM / img.height)
new_size = (int(img.width * scale), int(img.height * scale))
img_small = img.resize(new_size, Image.Resampling.BILINEAR)
print("Small size:", img_small.size)

boxes, probs = mtcnn.detect(img_small)
if boxes is not None and len(boxes) > 0:
    box = boxes[0]
    print("Detected small box:", box)
    # Scale box up
    box_large = [b / scale for b in box]
    print("Large box:", box_large)
else:
    print("No face detected on small image")

