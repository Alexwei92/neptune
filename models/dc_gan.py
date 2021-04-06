import torch
import torch.nn as nn

from .utils import weights_init

class Generator(nn.Module):
    """
    Generator
    """
    def __init__(self, z_dim, ngf, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (in_channels) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, ndf, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class DCGAN(nn.Module):
    """
    DCGAN Model
    """
    def __init__(self,
                input_dim,
                in_channels,
                z_dim,
                **kwargs):
        super().__init__()

        self.netG = Generator(z_dim, input_dim, in_channels)
        self.netD = Discriminator(input_dim, in_channels)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.z_dim = z_dim

    def generate(self, z):
        x_recon = self.netG(z)
        return x_recon

    def discriminate(self, x):
        label = self.netD(x)
        return label

    def forward(self, x):
        pass

    def sample(self, n_samples, device=torch.device('cuda:0')):
        z = torch.randn(n_samples, self.z_dim, 1, 1).to(device)
        samples = self.generate(z)
        return samples

    def get_latent_dim(self):
        return self.z_dim