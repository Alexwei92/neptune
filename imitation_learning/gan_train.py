import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class GANTrain():
    '''
    GAN Training Agent
    '''
    def __init__(self, netG, netD, n_z, learning_rate=2e-3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.n_z = n_z
        self.real_label = 1
        self.fake_label = 0
        self.criterion = nn.BCELoss()

        self.last_epoch = 0
        self.epoch = []
        self.errD = []
        self.errG = []

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.epoch = checkpoint['epoch']
        self.last_epoch = self.epoch[-1]
        self.errD = checkpoint['errD']
        self.errG = checkpoint['errG']

    # def loss_function(self, x_recon, x, mu, logvar):
    #     MSE = F.mse_loss(x_recon, x, reduction='sum')
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     beta = 5.0
    #     return MSE + beta * KLD, MSE, KLD

    def train(self, epoch, train_loader):
        self.netG.train()
        self.netD.train()
        netG_losses, netD_losses = 0.0, 0.0
        for batch_idx, batch_x in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            self.netD.zero_grad()
            batch_real = batch_x.to(self.device)
            batch_size = batch_real.size(0)
            label = torch.full((batch_size,), self.real_label,
                            dtype=batch_real.dtype, device=self.device)

            output = self.netD(batch_real)
            errD_real = self.criterion(output, label)
            errD_real.backward()
            # D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.n_z, 1, 1, device=self.device)
            batch_fake = self.netG(noise)
            label.fill_(self.fake_label)
            output = self.netD(batch_fake.detach())
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            netD_losses += errD.item()
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            output = self.netD(batch_fake)
            errG = self.criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            netG_losses += errG.item()
            self.optimizerG.step()
            
        N_total = len(train_loader.dataset)
        self.epoch.append(epoch + self.last_epoch)
        self.errD.append(netD_losses / N_total)
        self.errG.append(netG_losses / N_total)
        tqdm.write('Epoch: {:d}, Loss_D = {:.4f}, Loss_G = {:.4f}'.format(epoch + self.last_epoch, 
                    netD_losses / N_total, netG_losses / N_total))

    # def test(self, test_loader):
    #     self.VAE_model.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for batch_idx, batch_x in enumerate(test_loader):
    #             batch_x = batch_x.to(self.device)
    #             batch_x_recon, mu, logvar = self.VAE_model(batch_x)
    #             total_loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
    #             test_loss += total_loss.item()
    #     test_loss /= len(test_loader.dataset)
    #     print('Test set: Avg. loss: {:.4f}'.format(test_loss))

    def save_checkpoint(self, epoch, file_path):
        torch.save({
            'epoch': self.epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'errD': self.errD,
            'errG': self.errG,
        }, file_path)
        # print('Save checkpoint to ', file_path)

    def save_model(self, file_path):
        torch.save(self.netG.state_dict(), file_path)
        # print('Save model to ', file_path)

    def get_current_epoch(self):
        return self.epoch[-1]
