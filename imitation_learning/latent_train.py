import os
import torch
import torch.nn.functional as F
from torch import nn, optim

from utils import plot_train_losses

class LatentTrain():
    '''
    Latent Controller Training Agent
    '''
    def __init__(self, Latent_model, VAE_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Latent_model = Latent_model.to(self.device)
        self.VAE_model = VAE_model.to(self.device)
        self.optimizer = optim.Adam(self.Latent_model.parameters(), lr=1e-2)

        self.last_epoch = 0
        self.train_losses = []
        self.train_counter = []

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.Latent_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.last_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.train_counter = checkpoint['train_counter']

    def load_VAEmodel(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        model = torch.load(file_path)
        self.VAE_model.load_state_dict(model['model_state_dict'])

    def loss_function(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction='sum')
    
    def train(self, epoch, train_loader):
        epoch += self.last_epoch
        self.Latent_model.train()
        self.VAE_model.eval()
        # train_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_image, batch_label = batch_data
            batch_z = self.VAE_model.get_latent(batch_image.to(self.device))

            self.optimizer.zero_grad()
            batch_x_pred = self.Latent_model(batch_z).view(-1)
            loss = self.loss_function(batch_x_pred, batch_label.to(self.device))
            loss.backward()
            # train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3e}'.format(
                    epoch, batch_idx * len(batch_data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(batch_data)))
                self.train_losses.append(loss.item() / len(batch_data))
                self.train_counter.append(
                    (batch_idx*len(batch_data)) + ((epoch-1)*len(train_loader.dataset)))

    # def test(self, test_loader):
    #     self.Latent_model.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for batch_idx, batch_x in enumerate(test_loader):
    #             batch_x = batch_x.to(self.device)
    #             batch_x_pred = self.VAEmodel(batch_x)
    #             loss = self.loss_function(batch_x_recon, batch_x)
    #             test_loss += loss.item()
    #     test_loss /= len(test_loader.dataset)
    #     print('Test set: Avg. loss: {:.4f}'.format(test_loss))

    def save_checkpoint(self, epoch, file_path):
        torch.save({
            'epoch': epoch + self.last_epoch,
            'model_state_dict': self.Latent_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_counter': self.train_counter,
        }, file_path)
        print('Save checkpoint to {}'.format(file_path))

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.Latent_model.state_dict(),
        }, file_path)
        print('Save model to ', file_path)

    def get_train_history(self):
        return self.train_counter, self.train_losses

    def plot_train_result(self):
        plot_train_losses(self.get_train_history())