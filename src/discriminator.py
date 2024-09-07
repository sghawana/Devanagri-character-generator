import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_CNN(nn.Module):
    def __init__(self, activ, device, dtype):
        super().__init__()
        self.activ = activ
        self.device = device
        self.dtype = dtype
            
        ## out = floor[ (in-kernel)/stride   +   1]
        ## 32 --> 14 --> 5 --> Linear Layers
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=4, kernel_size=5, stride=2, device=self.device, dtype=self.dtype)
        self.conv2 = nn.Conv2D(in_channels=4, out_channels=8, kernel_size=5, stride=2, device=self.device, dtype=self.dtype)
        
        self.linear1 = nn.Linear(in_features=8*5*5, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=1) 

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
        x = x.reshape(x.shape[0], -1)
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = F.sigmoid(x)
        return x
        