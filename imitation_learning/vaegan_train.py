import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

import numpy as np

class VAEGANTrain():
    """
    VAE/GAN Training Agent
    """
    def __init__(self, VAEGAN_model, n_z, learning_rate=2e-3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VAEGAN_model = VAEGAN_model.to(self.device)
        self.optimizerE = optim.Adam(self.VAEGAN_model.netE.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.VAEGAN_model.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.VAEGAN_model.netD.parameters(), lr=learning_rate*0.1, betas=(0.5, 0.999))
        self.n_z = n_z

        self.last_epoch = 0
        self.epoch = []
        self.loss_E = []
        self.loss_G = []
        self.loss_D = []
        self.loss_kld = []

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.VAEGAN_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizerE.load_state_dict(checkpoint['optimizerE'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD'])
        self.epoch = checkpoint['epoch']
        self.last_epoch = self.epoch[-1]
        self.loss_E = checkpoint['loss_E']
        self.loss_G = checkpoint['loss_G']
        self.loss_D = checkpoint['loss_D']
        self.loss_kld = checkpoint['loss_kld']

    def train_batch(self, batch_x_real):
        batch_size = batch_x_real.size(0)
        y_real = torch.ones(batch_size).to(self.device)
        y_fake = torch.zeros(batch_size).to(self.device)

        # Extract fake images corresponding to real images
        mu, logvar = self.VAEGAN_model.netE(batch_x_real)
        batch_z = self.VAEGAN_model.reparameterize(mu, logvar)

        # Extract fake images corresponding to real images
        batch_x_fake = self.VAEGAN_model.netG(batch_z)

        # Extract latent_z corresponding to noise
        batch_z_prior = torch.randn(batch_size, self.n_z).to(self.device)
        batch_x_prior = self.VAEGAN_model.netG(batch_z_prior)

        # Compute D(x) for real and fake images along with their features
        l_r, _ = self.VAEGAN_model.netD(batch_x_real)
        l_f, _ = self.VAEGAN_model.netD(batch_x_fake)
        l_p, _ = self.VAEGAN_model.netD(batch_x_prior)

        # D training
        loss_D = F.binary_cross_entropy(l_r, y_real) \
                + 1.0*(F.binary_cross_entropy(l_f, y_fake) + F.binary_cross_entropy(l_p, y_fake))
        self.optimizerD.zero_grad()
        loss_D.backward(retain_graph=True)
        self.optimizerD.step()

        # G training
        l_r, s_r = self.VAEGAN_model.netD(batch_x_real)
        l_f, s_f = self.VAEGAN_model.netD(batch_x_fake)
        l_p, s_p = self.VAEGAN_model.netD(batch_x_prior)
        loss_D = F.binary_cross_entropy(l_r, y_real) \
                + 1.0*(F.binary_cross_entropy(l_f, y_fake) + F.binary_cross_entropy(l_p, y_fake))

        feature_loss = ((s_f - s_r) ** 2).mean()
        gamma = 15
        loss_G = gamma * feature_loss - loss_D
        self.optimizerG.zero_grad()
        loss_G.backward(retain_graph=True)
        self.optimizerG.step()
        
        # E training
        mu, logvar = self.VAEGAN_model.netE(batch_x_real)
        batch_z = self.VAEGAN_model.reparameterize(mu, logvar)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld.mean()
        batch_x_fake = self.VAEGAN_model.netG(batch_z)
        _, s_r = self.VAEGAN_model.netD(batch_x_real)
        _, s_f = self.VAEGAN_model.netD(batch_x_fake)
        feature_loss = ((s_f - s_r) ** 2).mean()

        beta = 5
        loss_E = kld + beta * feature_loss
        self.optimizerE.zero_grad()
        loss_E.backward(retain_graph=True)
        self.optimizerE.step()
    
        return loss_E.item(), loss_G.item(), loss_D.item(), kld.item()

    def train(self, epoch, train_loader):
        self.VAEGAN_model.train()

        T_loss_E = []
        T_loss_G = []
        T_loss_D = []
        T_loss_kld = []
        for batch_idx, batch_x, in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            loss_E, loss_G, loss_D, loss_kld = self.train_batch(batch_x)
            T_loss_D.append(loss_D)
            T_loss_G.append(loss_G)
            T_loss_E.append(loss_E)
            T_loss_kld.append(loss_kld)

        T_loss_D = np.mean(T_loss_D)
        T_loss_G = np.mean(T_loss_G)
        T_loss_E = np.mean(T_loss_E)
        T_loss_kld = np.mean(T_loss_kld)
        self.epoch.append(epoch + self.last_epoch)
        self.loss_D.append(T_loss_D)
        self.loss_G.append(T_loss_G)
        self.loss_E.append(T_loss_E)
        self.loss_kld.append(T_loss_kld)
        tqdm.write("epoch:{:d}, loss_E={:.4f}, loss_G={:.4f}, loss_D={:.4f}, loss_kld={:.4f}".format(epoch + self.last_epoch, T_loss_E, T_loss_G, T_loss_D, T_loss_kld))

    def test(self, test_data):
        pass

    def save_checkpoint(self, epoch, file_path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.VAEGAN_model.state_dict(),
            'optimizerE': self.optimizerE.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'loss_E': self.loss_E,
            'loss_G': self.loss_G,
            'loss_D': self.loss_D,
            'loss_kld': self.loss_kld,
        }, file_path)

    def save_model(self):
        pass

    def get_current_epoch(self):
        return self.epoch[-1]