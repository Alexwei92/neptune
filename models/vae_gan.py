import torch
from torch import nn
import torch.nn.functional as F

from .vanilla_vae import Encoder, Decoder
from .utils import weights_init

class Discriminator(nn.Module):
    """
    GAN Discriminator
    """

    def __init__(self, n_chan=3):
        super().__init__()

        self.linear0 = nn.Linear(512*4*4, 512)
        self.linear1 = nn.Linear(512, 1)

        self.convs = nn.Sequential(
            nn.Conv2d(n_chan, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )

        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        middle_feature = self.convs(x)      
        output = self.last_layer(middle_feature)
        return output.squeeze(), middle_feature.squeeze()

class VAEGAN(nn.Module):
    """
    VAEGAN Model
    """
    def __init__(self,
                input_dim,
                in_channels,
                z_dim,
                **kwargs):
        super().__init__()

        self.netE = Encoder(z_dim, in_channels)
        self.netG = Decoder(z_dim)
        self.netD = Discriminator(in_channels)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.z_dim = z_dim

    def encode(self, x):
        mu, logvar = self.netE(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, with_logvar=True):
        if with_logvar:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x_recon = self.netG(z)
        return x_recon

    def discriminate(self, x):
        label, similarity = self.netD(x)
        return label, similarity

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return [x_recon, x, mu, logvar]

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