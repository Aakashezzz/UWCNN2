import cv2
import os
import numpy as np
from tqdm import tqdm  #progress bar

clean_folder = "data/clean"
synthetic_folder = "data/synthetic"

os.makedirs(synthetic_folder, exist_ok=True)

def apply_underwater_physics_advanced(image):
    image = image.astype(np.float32) / 255.0
    h, w, _ = image.shape

    #Random attenuation coefficients (Beer-Lambert)(Light Absorption)
    beta_r = np.random.uniform(0.8, 1.5)
    beta_g = np.random.uniform(0.4, 1.0)
    beta_b = np.random.uniform(0.2, 0.6)

    #Simulated depth map (gradient + noise)
    depth = np.tile(np.linspace(0.2, 1.5, h).reshape(h, 1), (1, w))
    depth += np.random.normal(0, 0.05, (h, w))
    depth = np.clip(depth, 0.1, 2.0)

    # Transmission map per channel (Light+Depth)
    t_r = np.exp(-beta_r * depth)
    t_g = np.exp(-beta_g * depth)
    t_b = np.exp(-beta_b * depth)

    t = np.stack([t_b, t_g, t_r], axis=2)

    #Random backscatter(light Scattering)
    B = np.array([
        np.random.uniform(0.6, 0.9),  # Blue
        np.random.uniform(0.5, 0.8),  # Green
        np.random.uniform(0.3, 0.6)   # Red
    ])

    # Apply imaging model
    underwater = image * t + B * (1 - t)

    #Add sensor noise
    noise = np.random.normal(0, 0.02, underwater.shape) 
    underwater += noise

    underwater = np.clip(underwater, 0, 1)
    underwater = (underwater * 255).astype(np.uint8)

    return underwater


for filename in tqdm(os.listdir(clean_folder)):
    img_path = os.path.join(clean_folder, filename)
    clean_img = cv2.imread(img_path)

    if clean_img is None:
        continue

    clean_img = cv2.resize(clean_img, (256, 256))
    underwater_img = apply_underwater_physics_advanced(clean_img)

    cv2.imwrite(os.path.join(synthetic_folder, filename), underwater_img)

print("Improved synthetic dataset generated successfully.")