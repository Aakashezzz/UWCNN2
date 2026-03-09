import numpy as np
import torch

def generate_depth_map(height, width):
    depth = np.tile(np.linspace(0, 1, height), (width, 1)).T
    return depth

def apply_underwater_physics(clean_img):
    clean_img = clean_img.astype(np.float32) / 255.0
    h, w, c = clean_img.shape

    depth = generate_depth_map(h, w)

    # Random attenuation per channel (more realistic)
    beta_r = np.random.uniform(1.0, 1.8)
    beta_g = np.random.uniform(0.6, 1.2)
    beta_b = np.random.uniform(0.3, 0.8)

    transmission_r = np.exp(-beta_r * depth)
    transmission_g = np.exp(-beta_g * depth)
    transmission_b = np.exp(-beta_b * depth)

    transmission = np.stack([transmission_r,
                             transmission_g,
                             transmission_b], axis=2)

    # Stronger bluish backscatter
    B = np.array([
        np.random.uniform(0.05, 0.2),
        np.random.uniform(0.4, 0.7),
        np.random.uniform(0.7, 1.0)
    ])

    underwater_img = clean_img * transmission + B * (1 - transmission)
    underwater_img = np.clip(underwater_img, 0, 1)

    return (underwater_img * 255).astype(np.uint8)



def apply_physics_torch(clean_pred, depth):
    """
    clean_pred: [B,3,H,W]
    depth: [B,1,H,W]
    """
    device = clean_pred.device
    B, _, H, W = clean_pred.shape

    depth = depth.to(device)

    beta_r = torch.rand(B, 1, 1, 1, device=device) * 0.8 + 0.8   # 0.8–1.6
    beta_g = torch.rand(B, 1, 1, 1, device=device) * 0.5 + 0.5   # 0.5–1.0
    beta_b = torch.rand(B, 1, 1, 1, device=device) * 0.3 + 0.2   # 0.2–0.5

    transmission_r = torch.exp(-beta_r * depth)
    transmission_g = torch.exp(-beta_g * depth)
    transmission_b = torch.exp(-beta_b * depth)

    transmission = torch.cat([transmission_r,
                              transmission_g,
                              transmission_b], dim=1)

    B_light = torch.rand(B, 3, 1, 1, device=device) * torch.tensor(
        [0.3, 0.5, 0.7], device=device
    ).view(1,3,1,1)

    underwater_reconstructed = clean_pred * transmission + B_light * (1 - transmission)

    return underwater_reconstructed