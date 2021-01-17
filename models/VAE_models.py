import torch
from .NN_models import *

class MyVAE(nn.Module):

    def __init__(self, n_z=20):
        super().__init__()
        self.q_img = Dronet(n_chan=3, n_out=n_z*2)
        self.p_img = Decoder(n_in=n_z)
        self.n_z = n_z
        
    def encode(self, x):
        x = self.q_img(x)
        mu = x[:, :self.n_z].view(-1, self.n_z)
        logvar = x[:, self.n_z:].view(-1, self.n_z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.p_img(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar