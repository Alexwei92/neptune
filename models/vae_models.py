import torch
from torch import nn
import torch.nn.functional as F

# CNN output size
def calculate_output_size(n_in, kernel_size, stride=1, padding=0, dilation=1):
    n_out = (n_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(n_out)

def calculate_NN_size(input_size, kernel_size, stride=[1,1], padding=[0,0], dilation=[1,1]):
    H_in, W_in = input_size
    H_out = calculate_output_size(H_in, kernel_size[0], stride[0], padding[0], dilation[0])
    W_out = calculate_output_size(W_in, kernel_size[1], stride[1], padding[1], dilation[1])
    return (H_out, W_out)

class Dronet(nn.Module):
    """
    VAE Encoder
    """

    def __init__(self, n_chan, n_out):
        super().__init__()
        self.n_chan = n_chan
        self.n_out = n_out

        self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0 = nn.Conv2d(self.n_chan, 32, kernel_size=5, stride=2, padding=2)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)

        self.reshape = (-1, 512)
        self.linear0 = nn.Linear(512, 64)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, n_out)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)

    def forward(self, x):
        # Input
        x1 = self.conv0(x)
        x1 = self.max0(x1)

        # First residual block
        x2 = self.bn0(x1)
        # x2 = x1
        x2 = F.relu(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = torch.add(x1, x2)

        # Second residual block
        x4 = self.bn2(x3)
        # x4 = x3
        x4 = F.relu(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = F.relu(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = torch.add(x3, x4)

        # Third residual block
        x6 = self.bn4(x5)
        # x6 = x5
        x6 = F.relu(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = F.relu(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = torch.add(x5, x6)

        # Fully-connected layers
        x = torch.flatten(x7)
        x = x.view(self.reshape)

        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class Decoder(nn.Module):
    """
    VAE Decoder
    """

    def __init__(self, n_in):
        super().__init__()

        self.linear = nn.Linear(n_in, 1024)
        self.reshape = (-1, 1024, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, dilation=3)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=1, dilation=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1)
        # self.deconv6 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2)
        self.deconv7 = nn.ConvTranspose2d(16, 3, kernel_size=6, stride=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(self.reshape)

        x = self.deconv1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = F.relu(x)

        x = self.deconv3(x)
        x = F.relu(x)

        x = self.deconv4(x)
        x = F.relu(x)

        x = self.deconv5(x)
        x = F.relu(x)

        # x = self.deconv6(x)
        # x = F.relu(x)

        x = self.deconv7(x)
        x = torch.tanh(x)

        return x


class MyVAE(nn.Module):
    """
    Full VAE Model
    """

    def __init__(self, n_z=20):
        super().__init__()
        self.q_img = Dronet(n_chan=3, n_out=n_z * 2)
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

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
