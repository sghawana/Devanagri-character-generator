import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_CNN(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, device=self.device, dtype=self.dtype)
        self.bn1 = nn.BatchNorm2d(num_features=64, device=self.device, dtype=self.dtype)
        self.d1 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, device=self.device, dtype=self.dtype)
        self.bn2 = nn.BatchNorm2d(num_features=128, device=self.device, dtype=self.dtype)
        self.d2 = nn.Dropout2d(p=0.2)
        
        self.linear1 = nn.Linear(in_features=128*5*5, out_features=100, device=self.device, dtype=self.dtype)
        self.bn3 = nn.BatchNorm1d(num_features=100, device=self.device, dtype=self.dtype)
        self.d3 = nn.Dropout1d(p=0.2)
        
        self.linear2 = nn.Linear(in_features=100, out_features=1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.d1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.d2(x)

        x = x.reshape(-1, 128*5*5)

        x = self.linear1(x)
        x = x = F.tanh(x)

        x = self.linear2(x)
        x = F.sigmoid(x)
        return x