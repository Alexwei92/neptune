import torch 
from torch import nn
import torch.nn.functional as F

class NN(nn.Module):
    '''
    Simple fully-connected network for IL
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        self.linear5 = nn.Linear(1000, 1000)
        self.linear6 = nn.Linear(1000, 1)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(1000)

    def forward(self, x):
        # Hidden layers
        x = self.linear1(x)
        # x = self.bn1(x)
        x = F.relu(x)

        x = self.linear2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.linear3(x)
        # x = self.bn3(x)
        x = F.relu(x)

        x = self.linear4(x)
        # x = self.bn3(x)
        x = F.relu(x)

        x = self.linear5(x)
        # x = self.bn3(x)
        x = F.relu(x)

        # Output layer
        x = self.linear6(x)
        x = torch.tanh(x)

        return x

class LatentNN_complex(nn.Module):
    """
    Latent NN Model
    """
    def __init__(self,
                z_dim,
                num_prvs,
                **kwargs):
        super().__init__()
        
        self.NN = NN(z_dim + num_prvs + 1)
        self.z_dim = z_dim

    def forward(self, x):
        y_pred = self.NN(x)
        return y_pred

    def loss_function(self, y_pred, y):
        batch_size = y.size(0)
        loss = F.mse_loss(y_pred, y, reduction='sum').div(batch_size)        
        return {'total_loss': loss}

    def get_latent_dim(self):
        return self.z_dim