import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UWCNN

model = UWCNN()
model.load_state_dict(torch.load("2uwcnn_level2_model.pth", map_location=torch.device('cpu')))
model.eval()

test_folder = "data/test"

img_name = "370_img_.png"   # <<< TYPE IMAGE NAME HERE
img_path = os.path.join(test_folder, img_name)

if not os.path.exists(img_path):
    print("Image not found:", img_name)
    exit()

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(
    img_rgb.astype("float32") / 255.0
).permute(2, 0, 1).unsqueeze(0)


with torch.no_grad():
    output = model(img_tensor)

output_img = output.squeeze().permute(1, 2, 0).numpy()
output_img = np.clip(output_img, 0, 1)

plt.figure(figsize=(5, 3))

plt.suptitle(f"Image: {img_name}", fontsize=14)

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Enhanced")
plt.imshow(output_img)
plt.axis("off")

plt.tight_layout()
plt.show()

"""
import torch
import cv2
import numpy as np
import os
from model import UWCNN

# ==============================
# SETTINGS
# ==============================

model_path = "2uwcnn_level2_model.pth"
test_folder = "data/test"
output_folder = "data/comparison"

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
# PROCESS ALL IMAGES
# ==============================

img_list = sorted(os.listdir(test_folder))

for img_name in img_list:

    img_path = os.path.join(test_folder, img_name)
    original_bgr = cv2.imread(img_path)

    if original_bgr is None:
        print(f"Skipping {img_name}")
        continue

    original_bgr = cv2.resize(original_bgr, (256, 256))

    # Convert to RGB for model
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = torch.tensor(
        original_rgb.astype("float32") / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    enhanced = output.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced, 0, 1)
    enhanced = (enhanced * 255).astype(np.uint8)

    # Convert enhanced back to BGR for saving
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # ==============================
    # STITCH SIDE-BY-SIDE
    # ==============================

    comparison = np.hstack((original_bgr, enhanced_bgr))

    # Optional: Add labels
    cv2.putText(comparison, "Original", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.putText(comparison, "Enhanced", (276, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, comparison)

    print(f"Saved comparison: {img_name}")

print("All comparison images saved in:", output_folder)


"""











""""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UWCNN

model = UWCNN()
model.load_state_dict(torch.load("2uwcnn_level2_model.pth", map_location=torch.device('cpu')))
##model.load_state_dict(torch.load("uwcnn_level2_model.pth", map_location=torch.device('cpu')))
model.eval()

test_folder = "data/test"
img_name = os.listdir(test_folder)[1]
img_path = os.path.join(test_folder, img_name)

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(img_rgb.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)

output_img = output.squeeze().permute(1, 2, 0).numpy()
output_img = np.clip(output_img, 0, 1)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Enhanced")
plt.imshow(output_img)
plt.axis("off")

plt.show()
"""