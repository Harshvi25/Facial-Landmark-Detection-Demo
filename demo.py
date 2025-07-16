# demo.py

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from model import MultiStageCNN     
from data import get_transforms     

IMAGE_PATH = "sample_image.jpg"                # Replace this with your demo image
CHECKPOINT_PATH = "best_heatmap_model_2.pth"   # Trained model weights
SAVE_DIR = "outputs_demo"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

model = MultiStageCNN(stages=5).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(original_img, (256, 256))
pil_img = Image.fromarray(original_img)

transform = get_transforms()
input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    heatmaps = model(input_tensor)[-1].squeeze(0).cpu().numpy()  # (68, 128, 128)

coords = []
for hmap in heatmaps:
    y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
    coords.append((x * 2, y * 2))  # upscale to 256x256

output_img = resized_img.copy()
for x, y in coords:
    cv2.circle(output_img, (int(x), int(y)), 2, (0, 255, 0), -1)

save_path = os.path.join(SAVE_DIR, "predicted_landmarks.jpg")
cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
plt.imshow(output_img)
plt.title("Predicted Facial Landmarks")
plt.axis("off")
plt.show()

print(f"\n Saved output image with landmarks to: {save_path}")
