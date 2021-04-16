import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .vanilla_vae_old import Dronet, Decoder

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class BetaVAE_B(nn.Module):
    """
    Beta (B) VAE Model  
    """
    def __init__(self,
                input_dim,
                in_channels,
                z_dim,
                gamma,
                C_max,
                C_stop_iter,
                **kwargs):
        super().__init__()

        self.Encoder = Dronet(input_dim, z_dim, in_channels)
        self.Decoder = Decoder(z_dim)
        self.Decoder.apply(weights_init)
        self.z_dim = z_dim
        self.C_max = torch.Tensor([C_max])
        self.C_stop_iter = C_stop_iter
        self.gamma = gamma

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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return [x_recon, x, mu, logvar]

    def loss_function(self, *args, **kwargs):
        x_recon = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]

        batch_size = x_recon.size(0)
        mse = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_z = kld.mean(0)
        kld = kld.sum(1).mean(0)
        self.C_max = self.C_max.to(x_recon.device)
        C = torch.clamp(self.C_max / self.C_stop_iter * kwargs['num_iter'], 0, self.C_max.data[0])
        total_loss = mse + self.gamma * (kld - C).abs()
        # tqdm.write('MSE: {:.3e}, KLD: {:.3e}'.format(MSE, (self.gamma * (KLD - C).abs()).item()))

        return {'total_loss': total_loss,
                'mse_loss': mse,
                'kld_loss': kld,
                'kld_loss_z': kld_z
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