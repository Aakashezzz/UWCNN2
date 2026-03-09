import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + 32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 64, 32, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(torch.cat([x, out1], dim=1)))
        out3 = self.relu(self.conv3(torch.cat([x, out1, out2], dim=1)))

        return torch.cat([x, out1, out2, out3], dim=1)


class UWCNN(nn.Module):
    def __init__(self):
        super(UWCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.dense1 = DenseBlock(32)
        self.dense2 = DenseBlock(32 + 96)

        self.conv_out = nn.Conv2d(32 + 96 + 96, 3, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dense1(out)
        out = self.dense2(out)

        residual = self.conv_out(out)
        return torch.clamp(x + residual, 0, 1)