import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_CNN(nn.Module):
    def __init__(self, activ, device, dtype):
        super().__init__()
        self.activ = activ
        self.device = device
        self.dtype = dtype
        
        ## out = (in-1)*stride + kernel
        ## 3 --> 7 --> 15 --> 22 --> 28+4
        self.conv1 = nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, device=self.device, dtype=self.dtype)
        self.conv2 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, device=self.device, dtype=self.dtype)
        self.conv3 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=7, stride=1, device=self.device, dtype=self.dtype)
        self.conv4 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, output_padding=2, device=self.device, dtype=self.dtype)

    def activation(self, x):
        if self.activ == 'relu':
            return F.relu(x)
        elif self.activ == 'sigmoid':
            return F.sigmoid(x)
        elif self.activ == 'tanh':
            return F.tanh(x)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        return x
        
        
