import os
import torch
import torch.nn.functional as F
from torch import nn, optim

from utils import plot_generate_figure, plot_train_losses

class VAETrain():
    '''
    VAE Training Agent
    '''
    def __init__(self, VAE_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VAE_model = VAE_model.to(self.device)
        self.optimizer = optim.Adam(self.VAE_model.parameters(), lr=1e-3)

        self.last_epoch = 0
        self.train_losses = []
        self.train_MSE_losses = []
        self.train_KLD_losses = []
        self.train_counter = []

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.VAE_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.last_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.train_MSE_losses = checkpoint['train_MSE_losses']
        self.train_KLD_losses = checkpoint['train_KLD_losses']
        self.train_counter = checkpoint['train_counter']

    def loss_function(self, x_recon, x, mu, logvar):
        MSE = F.mse_loss(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = 100.0
        return MSE + beta * KLD, MSE, KLD

    def train(self, epoch, train_loader):
        epoch += self.last_epoch
        self.VAE_model.train()
        # train_loss = 0
        for batch_idx, batch_x in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            batch_x_recon, mu, logvar = self.VAE_model(batch_x)
            loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
            loss.backward()
            # train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch_x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(batch_x)))
                self.train_losses.append(loss.item() / len(batch_x))
                self.train_MSE_losses.append(MSE_loss.item() / len(batch_x))
                self.train_KLD_losses.append(KLD_loss.item() / len(batch_x))
                self.train_counter.append(
                    (batch_idx*len(batch_x)) + ((epoch-1)*len(train_loader.dataset)))

    def test(self, test_loader):
        self.VAE_model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                batch_x_recon, mu, logvar = self.VAE_model(batch_x)
                loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        print('Test set: Avg. loss: {:.4f}'.format(test_loss))

    def save_checkpoint(self, epoch, file_path):
        torch.save({
            'epoch': epoch + self.last_epoch,
            'model_state_dict': self.VAE_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_MSE_losses': self.train_MSE_losses,
            'train_KLD_losses': self.train_KLD_losses,
            'train_counter': self.train_counter,
        }, file_path)
        print('Save checkpoint to ', file_path)

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.VAE_model.state_dict(),
        }, file_path)
        print('Save model to ', file_path)

    def get_train_history(self):
        return self.train_counter, self.train_losses, self.train_MSE_losses, self.train_KLD_losses

    def get_latent(self, x):
        z = self.VAE_model.get_latent(x)
        return z

    def plot_generate_result(self, data_loader, N=6):
        examples = enumerate(data_loader)
        batch_idx, example_data = next(examples)
        with torch.no_grad():
            generated_data, _, _ = self.VAE_model(example_data.to(self.device))
            plot_generate_figure(generated_data.cpu(), example_data, N)

    def plot_train_result(self):
        plot_train_losses(self.get_train_history())