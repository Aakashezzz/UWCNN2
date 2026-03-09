import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels + 32, 32, 3, padding=1)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = torch.cat([x, out1], dim=1)
        out3 = self.relu(self.conv2(out2))
        return torch.cat([out2, out3], dim=1)

class UWCNN(nn.Module):
    def __init__(self):
        super(UWCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.dense1 = DenseBlock(32)
        self.conv2 = nn.Conv2d(32 + 64, 3, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dense1(out)
        residual = self.conv2(out)
        return torch.clamp(x + residual, 0, 1)