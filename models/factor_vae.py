import torch
from torch import nn
import torch.nn.functional as F

from .vanilla_vae_old import Dronet, Decoder
from .utils import weights_init

class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            # nn.ReLU(),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class FactorVAE(nn.Module):
    """
    Factor VAE Model
    """
    def __init__(self,
                input_dim,
                in_channels,
                z_dim,
                gamma,
                **kwargs):
        super().__init__()
        
        self.Encoder = Dronet(input_dim, z_dim, in_channels)
        self.Decoder = Decoder(z_dim)
        self.netD = Discriminator(z_dim)
        self.netD.apply(weights_init)
        self.z_dim = z_dim

        self.gamma = gamma
        self.D_z = None

    def encode(self, x):
        mu, logvar = self.Encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, with_logvar=True):
        if with_logvar:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x_recon = self.Decoder(z)
        return x_recon

    def discriminate(self, z):
        y = self.netD(z)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return [x_recon, x, mu, logvar, z]

    def permute_dims(self, z):
        perm_z = []
        batch_size = z.size(0)
        for z_j in z.split(1, dim=1):
            perm = torch.randperm(batch_size).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)
        return torch.cat(perm_z, axis=1)

    def loss_function(self, *args, **kwargs):
        x_recon = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]
        z = args[4]

        batch_size = x_recon.size(0)
        if kwargs['mode'] == 'VAE':
            mse = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kld_z = kld.mean(0)
            kld = kld.sum(1).mean(0)
            self.D_z = self.discriminate(z)
            # log[(D(z) / (1 - D(z)))] = log(D(z)) - log(1-D(z))
            vae_tc_loss = (self.D_z[:,0] - self.D_z[:,1]).mean()
            total_loss = kld + mse + self.gamma * vae_tc_loss
            return {
                'total_loss': total_loss,
                'mse_loss': mse,
                'kld_loss': kld,
                'kld_loss_z': kld_z,
                'vae_tc_loss': vae_tc_loss
            }
        
        elif kwargs['mode'] == 'netD':
            index_zeros = torch.zeros(batch_size, dtype=torch.long).to(x.device)
            index_ones = torch.ones(batch_size, dtype=torch.long).to(x.device)
            D_z = self.discriminate(z.detach())
            z_perm = self.permute_dims(z)
            D_z_perm = self.discriminate(z_perm.detach())
            D_tc_loss = 0.5 * (F.cross_entropy(D_z, index_zeros) +
                            F.cross_entropy(D_z_perm, index_ones))

            return {
                'D_tc_loss': D_tc_loss
            }

    def sample(self, n_samples, device=torch.device('cuda:0')):
        z = torch.randn(n_samples, self.z_dim).to(device)
        samples = self.decode(z)
        return samples

    def get_latent(self, x, with_logvar=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, with_logvar)
        return z

    def get_latent_dim(self):
        return self.z_dim