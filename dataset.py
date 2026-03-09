#provides data for training the model ie. clean and syntethic data
import os
import cv2
import torch
from torch.utils.data import Dataset

class UnderwaterDataset(Dataset):
    def __init__(self, clean_dir, synthetic_dir):
        self.clean_dir = clean_dir
        self.synthetic_dir = synthetic_dir
        self.image_names = os.listdir(clean_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        clean_path = os.path.join(self.clean_dir, img_name)
        synthetic_path = os.path.join(self.synthetic_dir, img_name)

        clean = cv2.imread(clean_path)
        synthetic = cv2.imread(synthetic_path)

        clean = cv2.resize(clean, (256, 256))
        synthetic = cv2.resize(synthetic, (256, 256))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        synthetic = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)

        clean = clean.astype("float32") / 255.0
        synthetic = synthetic.astype("float32") / 255.0

        clean = torch.tensor(clean).permute(2, 0, 1)
        synthetic = torch.tensor(synthetic).permute(2, 0, 1)

        return synthetic, clean
    

