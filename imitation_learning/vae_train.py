import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

class VAETrain():
    '''
    VAE Training Agent
    '''
    def __init__(self, VAEmodel):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VAEmodel = VAEmodel.to(self.device)
        self.optimizer = optim.Adam(self.VAEmodel.parameters(), lr=1e-3)

        self.last_epoch = 0
        self.train_losses = []
        self.train_MSE_losses = []
        self.train_KLD_losses = []
        self.train_counter = []

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.VAEmodel.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.last_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.train_MSE_losses = checkpoint['train_MSE_losses']
        self.train_KLD_losses = checkpoint['train_KLD_losses']
        self.train_counter = checkpoint['train_counter']

    def loss_function(self, x_recon, x, mu, logvar):
        MSE = F.mse_loss(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = 8.0
        return MSE + beta * KLD, MSE, KLD

    def train(self, epoch, train_loader):
        epoch += self.last_epoch
        self.VAEmodel.train()
        # train_loss = 0
        for batch_idx, batch_x in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            batch_x_recon, mu, logvar = self.VAEmodel(batch_x)
            loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
            loss.backward()
            # train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch_x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(batch_x)))
                self.train_losses.append(loss.item() / len(batch_x))
                self.train_MSE_losses.append(MSE_loss.item() / len(batch_x))
                self.train_KLD_losses.append(KLD_loss.item() / len(batch_x))
                self.train_counter.append(
                    (batch_idx*len(batch_x)) + ((epoch-1)*len(train_loader.dataset)))

    def test(self, test_loader):
        self.VAEmodel.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                batch_x_recon, mu, logvar = self.VAEmodel(batch_x)
                loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))

    def save_checkpoint(self, epoch, file_path):
        torch.save({
            'epoch': epoch + self.last_epoch,
            'model_state_dict': self.VAEmodel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_MSE_losses': self.train_MSE_losses,
            'train_KLD_losses': self.train_KLD_losses,
            'train_counter': self.train_counter,
        }, file_path)
        print('Save to checkpoint {}'.format(file_path))

    def get_train_history(self):
        return self.train_counter, self.train_losses, self.train_MSE_losses, self.train_KLD_losses

    def get_latent(self, x):
        z = self.VAEmodel.get_latent(x)
        return z
