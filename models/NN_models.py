import torch 
from torch import nn
import torch.nn.functional as F


class MyNN(nn.Module):
    '''
    Simple fully-connected network for IL
    '''
    def __init__(self, n_in):
        super().__init__()
        self.linear1 = nn.Linear(n_in, 256)
        self.linear2 = nn.Linear(256, 16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, x):
        # Hidden layers
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        # Output layer
        x = self.linear3(x)

        return x