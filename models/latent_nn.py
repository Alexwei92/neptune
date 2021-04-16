import torch 
from torch import nn
import torch.nn.functional as F

from .utils import weights_init

class NN(nn.Module):
    '''
    Simple fully-connected network for IL
    '''
    def __init__(self, z_dim, extra_dim=0):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)
        if extra_dim > 0:
            self.linear5 = nn.Linear(extra_dim + 1, 1)

    def forward(self, x, x_extra):
        # Hidden layers
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        x = self.linear4(x)
        x = torch.tanh(x)

        if x_extra is not None:
            x = torch.cat([x, x_extra], 1)
            x = self.linear5(x)
        
        return x

class NN2(nn.Module):
    '''
    Simple fully-connected network for IL
    '''
    def __init__(self, z_dim, with_yawRate=True):
        super().__init__()
        if with_yawRate:
            self.linear1 = nn.Linear(z_dim+1, 512)
        else:
            self.linear1 = nn.Linear(z_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)

    def forward(self, x):
        # Hidden layers
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        x = self.linear4(x)
        x = torch.tanh(x)
        
        return x

class LatentNN(nn.Module):
    """
    Latent NN Model
    """
    def __init__(self,
                z_dim,
                # num_prvs,
                with_yawRate,
                **kwargs):
        super().__init__()
        
        self.NN = NN2(z_dim, with_yawRate)
        self.NN.apply(weights_init)
        self.z_dim = z_dim

    def forward(self, x):
        y_pred = self.NN(x)
        return y_pred

    def loss_function(self, y_pred, y):
        loss = F.mse_loss(y_pred, y, reduction='mean')    
        return {'total_loss': loss}

    def get_latent_dim(self):
        return self.z_dim