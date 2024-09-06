import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_CNN(nn.Module):
    def __init__(self, activation, device, dtype, output_padding):
        super().__init__()
        self.activation = activation
        self.device = device
        self.dtype = dtype
        self.output_padding = output_padding
        
        ## out = (in-1)*stride + kernel
        ## 3 --> 7 --> 15 --> 22 --> 28/32
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2D(in_channels=4, out_channels=4, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2D(in_channels=4, out_channels=2, kernel_size=7, stride=1)
        self.conv4 = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, output_padding=self.output_padding)

    def activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return F.sigmoid(x)
        elif self.activation == 'tanh':
            return F.tanh(x)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x
        
        
