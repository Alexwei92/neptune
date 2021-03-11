import torch
from torch import nn
import torch.nn.functional as F

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    """
    VAE Encoder
    """

    def __init__(self, input_dim, n_out, n_chan=3):
        super().__init__()

        # Assume input has the size [n_chan, 64, 64]
        # 1) Very first CNN and MaxPool -> [32, 16, 16]
        self.conv0 = nn.Conv2d(n_chan, 32, kernel_size=5, stride=2, padding=2)
        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2) First residual block -> [32, 8, 8]
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)
        # 3) Second residual block -> [64, 4, 4]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        # 4) Third residual block -> [128, 2, 2]
        self.bn4 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        # 5) Linear layers -> [n_out]
        tmp_dim = int(input_dim / 16 * 128)
        self.reshape = (-1, tmp_dim)
        self.linear0 = nn.Linear(tmp_dim, n_out)
        self.linear1 = nn.Linear(tmp_dim, n_out)
        # Init the weight
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)

    def reparameterize(self, x):
        mu = x[:, :self.n_z].view(-1, self.n_z)
        logvar = x[:, self.n_z:].view(-1, self.n_z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return z, kld

    def forward(self, x):
        # 1) Input
        x1 = self.conv0(x)
        x1 = self.max0(x1)
        
        # 2) First residual block
        x2 = self.bn0(x1)
        x2 = F.relu(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = torch.add(x1, x2)

        # 3) Second residual block
        x4 = self.bn2(x3)
        x4 = F.relu(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = F.relu(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = torch.add(x3, x4)

        # 4) Third residual block
        x6 = self.bn4(x5)
        x6 = F.relu(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = F.relu(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = torch.add(x5, x6)

        # 5) Fully-connected layers
        x = torch.flatten(x7)
        x = x.view(self.reshape)

        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        mu = self.linear0(x)
        logvar = self.linear1(x)

        return mu, logvar

class Generator(nn.Module):
    """
    VAE Decoder / GAN Generator
    """

    def __init__(self, n_in):
        super().__init__()
        
        # 1) Linear layers
        self.linear0 = nn.Linear(n_in, 512)
        self.reshape = (-1, 512, 1, 1)

        # 2) Transposed convolutional layers
        self.deconv0 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=1, padding=0)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn0 = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(8)

    def forward(self, x):
        # 1) Linear layer
        x = self.linear0(x)
        x = x.view(self.reshape)

        # 2) Transposed convolutional layers
        x = self.deconv0(x)
        x = F.leaky_relu(x)
        x = self.bn0(x)

        x = self.deconv1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x)
        x = self.bn3(x)

        x = self.deconv4(x)
        x = F.leaky_relu(x)
        x = self.bn4(x)

        x = self.deconv5(x)
        x = torch.tanh(x)

        return x

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

        self.convs = nn.Sequential(
            # input is (nc) x 64 x 64
            self.conv0,
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 32 x 32
            self.conv1,
            self.bn1,
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 16 x 16
            self.conv2,
            self.bn2,
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 8 x 8
            self.conv3,
            self.bn3,
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.last_layer = nn.Sequential(
            self.conv4,
            nn.Sigmoid()
        )

    def forward(self, x):
        middle_feature = self.convs(x)
        output = self.last_layer(middle_feature)
        return output.squeeze(), middle_feature.squeeze()

    # def similarity(self, x):
    #     f_d = self.convs(x)
    #     return f_d.squeeze()

class MyVAEGAN(nn.Module):
    """
    Full VAE/GAN Model
    """

    def __init__(self, input_dim, n_z=20, n_chan=3):
        super().__init__()
        self.netE = Encoder(input_dim, n_out=n_z, n_chan=3)
        self.netG = Generator(n_z)
        self.netD = Discriminator()
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.n_z = n_z

    def encode(self, x):
        mu, logvar = self.netE(x)
        z = self.reparameterize(mu, logvar)
        return z

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
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    # def get_latent(self, x):
    #     z = self.encode(x)
    #     return z
