import torch
from torch import nn
import torch.nn.functional as F

from .vanilla_vae import Dronet, Decoder
from .utils import weights_init

class Discriminator(nn.Module):
    """
    GAN Discriminator
    """

    def __init__(self, n_chan=3):
        super().__init__()

        self.conv0 = nn.Conv2d(n_chan, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm1d(512)

        self.linear0 = nn.Linear(512*4*4, 512)
        self.linear1 = nn.Linear(512, 1)

        self.convs = nn.Sequential(
            # input is (nc) x 64 x 64
            self.conv0,
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            # state size. (ndf) x 32 x 32
            self.conv1,
            self.bn1,
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            # state size. (ndf*2) x 16 x 16
            self.conv2,
            self.bn2,
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            # state size. (ndf*4) x 8 x 8
            self.conv3,
            self.bn3,
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU()
        )

        self.last_layer = nn.Sequential(
            self.linear0,
            self.bn4,
            nn.ReLU(),
            self.linear1,
            # self.conv4,
            nn.Sigmoid()
        )

    def forward(self, x):
        middle_feature = self.convs(x)
        middle_feature = middle_feature.view(-1, 512*4*4)
        output = self.last_layer(middle_feature)
        return output.squeeze(), middle_feature.squeeze()

    # def similarity(self, x):
    #     f_d = self.convs(x)
    #     return f_d.squeeze()

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

        self.netE = Dronet(input_dim, z_dim, in_channels)
        self.netG = Decoder(z_dim)
        self.netD = Discriminator()
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.z_dim = z_dim

    def encode(self, x):
        mu, logvar = self.netE(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        return x_recon, mu, logvar

    def sample(self, n_samples, device=torch.device('cuda:0')):
        z = torch.randn(n_samples, self.z_dim).to(device)
        samples = self.decode(z)
        return samples

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def get_latent_dim(self):
        return self.z_dim