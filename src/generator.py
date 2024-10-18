import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_CNN(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.project = nn.Linear(in_features=100, out_features=1024*4*4, device=self.device, dtype=self.dtype)

        self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1, device=self.device, dtype=self.dtype)
        self.bn1 = nn.BatchNorm2d(num_features=256, device=self.device, dtype=self.dtype)
        self.d1 = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, device=self.device, dtype=self.dtype)
        self.bn2 = nn.BatchNorm2d(num_features=32, device=self.device, dtype=self.dtype)
        self.d2 = nn.Dropout2d(p=0.2)
        
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, device=self.device, dtype=self.dtype)
        self.bn3 = nn.BatchNorm2d(num_features=32, device=self.device, dtype=self.dtype)
        self.d3 = nn.Dropout2d(p=0.2)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, device=self.device, dtype=self.dtype)
        
    def forward(self, x):
        
        x = self.project(x)
        x = F.tanh(x)
        x = x.reshape(-1, 1024, 4, 4)
        
        x = self.conv1(x, output_size=(-1, 512, 8, 8))
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.d1(x)
        
        x = self.conv2(x, output_size=(-1, 256, 16, 16))
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.d2(x)
        
        x = self.conv3(x, output_size=(-1, 128, 32, 32))
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.d3(x)
        
        x = self.conv4(x)
        x = F.sigmoid(x)
        
        return x

        
        
