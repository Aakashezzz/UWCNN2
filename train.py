import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UnderwaterDataset
from model import UWCNN
from physics_model import apply_physics_torch
from pytorch_msssim import ssim
from tqdm import tqdm
import numpy as np
import os

clean_dir = "data/clean"
synthetic_dir = "data/synthetic"

batch_size = 4
epochs = 50
learning_rate = 0.001
lambda_physics = 0.5
lambda_ssim = 0.2

device = torch.device("cpu")

dataset = UnderwaterDataset(clean_dir, synthetic_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UWCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training started...")
print("Number of training images:", len(dataset))
print("Batch size:", batch_size)
print("Batches per epoch:", len(dataloader))
print("--------------------------------------------------")


for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_mse = 0
    total_ssim = 0
    total_phys = 0

    for batch_idx, (synthetic, clean) in enumerate(tqdm(dataloader)):

        synthetic = synthetic.to(device)
        clean = clean.to(device)

    
        if epoch == 0 and batch_idx == 0:
            print("Synthetic Min:", synthetic.min().item())
            print("Synthetic Max:", synthetic.max().item())
            print("Clean Min:", clean.min().item())
            print("Clean Max:", clean.max().item())
            print("--------------------------------------------------")

        optimizer.zero_grad()

      
        output = model(synthetic)

     
        mse_loss = nn.functional.mse_loss(output, clean)

        ssim_loss = 1 - ssim(output, clean, data_range=1, size_average=True)

      
        B, _, H, W = synthetic.shape

        depth = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
        depth = depth + 0.05 * torch.randn_like(depth)
        depth = torch.clamp(depth, 0, 1).to(device)

        reconstructed_input = apply_physics_torch(output, depth)

        physics_loss = nn.functional.mse_loss(reconstructed_input, synthetic)

        loss = mse_loss + lambda_ssim * ssim_loss + lambda_physics * physics_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_ssim += ssim_loss.item()
        total_phys += physics_loss.item()

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Total Loss: {total_loss/len(dataloader):.6f}")
    print(f"MSE Loss: {total_mse/len(dataloader):.6f}")
    print(f"SSIM Loss: {total_ssim/len(dataloader):.6f}")
    print(f"Physics Loss: {total_phys/len(dataloader):.6f}")
    print("--------------------------------------------------")

torch.save(model.state_dict(), "2uwcnn_level2_model.pth")
print("Training completed and model saved as uwcnn_level2_model.pth")