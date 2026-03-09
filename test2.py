import torch
import cv2
import numpy as np
import os
from model2 import UWCNN

# ==============================
# SETTINGS
# ==============================

model_path = "uwcnn_level2_model.pth"
test_folder = "data/test"
output_folder = "data/output"

os.makedirs(output_folder, exist_ok=True)

device = torch.device("cpu")

# ==============================
# LOAD MODEL
# ==============================

model = UWCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully.")

# ==============================
# LOAD IMAGE LIST
# ==============================

img_list = sorted(os.listdir(test_folder))

if len(img_list) == 0:
    print("No test images found.")
    exit()

print("Total test images:", len(img_list))
print("--------------------------------------------------")

# ==============================
# PROCESS ALL IMAGES
# ==============================

for img_name in img_list:

    img_path = os.path.join(test_folder, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Skipping {img_name} (failed to load)")
        continue

    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0

    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = np.clip(output_image, 0, 1)
    output_image = (output_image * 255).astype(np.uint8)

    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, output_image)

    print(f"Processed: {img_name}")

print("--------------------------------------------------")
print("All images enhanced and saved in:", output_folder)