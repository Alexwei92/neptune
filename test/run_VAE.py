import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import numpy as np

import setup_path
from utils import *
from models import *

class MyAgent(object):
    def __init__(self, disp_interval=10):
        self.optimizer = optim.Adam(VAEmodel.parameters(), lr=learning_rate)
        self.disp_interval = disp_interval
        
        if flag['load_checkpoint']:
            checkpoint = torch.load(checkpoint_path)
            VAEmodel.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.train_losses = checkpoint['train_losses']
            self.train_counter = checkpoint['train_counter']
        else:
            self.last_epoch = 0
            self.train_losses = []
            self.train_counter = []

    def loss_function(self, x_recon, x, mu, logvar):
        MSE = F.mse_loss(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = 8.0
        return MSE + beta * KLD, MSE, KLD

    def train(self, epoch):
        epoch += self.last_epoch
        VAEmodel.train()
        # train_loss = 0
        for batch_idx, batch_x in enumerate(train_loader):
            batch_x = batch_x.to(device)
            self.optimizer.zero_grad()
            batch_x_recon, mu, logvar = VAEmodel(batch_x)
            loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
            loss.backward()
            # train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.disp_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch_x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(batch_x)))
                self.train_losses.append(loss.item() / len(batch_x))
                self.train_counter.append(
                    (batch_idx*len(batch_x)) + ((epoch-1)*len(train_loader.dataset)))

        if flag['save_checkpoint']:
            if epoch % 5 == 0 and epoch > 0:
                print('Save to checkpoint {}'.format(checkpoint_path))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': VAEmodel.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'train_counter': self.train_counter,
                }, checkpoint_path)

    def test(self):
        VAEmodel.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(test_loader):
                batch_x = batch_x.to(device)
                batch_x_recon, mu, logvar = VAEmodel(batch_x)
                loss, MSE_loss, KLD_loss = self.loss_function(batch_x_recon, batch_x, mu, logvar)
                test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))


    def get_train_history(self):
        return self.train_losses, self.train_counter


if __name__ == '__main__':

    # Define data and output folder directories
    data_dir = os.path.join(setup_path.parent_dir, 'my_datasets/images_1k')
    output_dir = os.path.join(setup_path.parent_dir,'my_outputs/')
    checkpoint_path = os.path.join(output_dir, 'checkpoint_gate_z_10.tar')
    if not os.path.isdir(data_dir):
        raise IOError("No such data folder!")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Parameters
    flag = {
        'plot_samples': True,
        'plot_loss': True,
        'resume_training': True,
        'load_checkpoint': False,
        'save_checkpoint': False,
    }

    batch_size = 32
    img_res = (64,64)
    learning_rate=1e-4
    n_epochs = 5
    z_dim = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True

    # DataLoader
    # Option 1: PIL image
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ]) 
    # all_data = MyDataset_PIL(data_dir, transform=transform)

    # Option 2: numpy.array image
    all_data = MyDataset_np(data_dir, res=img_res, n_chan=3)
    train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=11) # split into train and test datasets

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    print('===> Loaded the datasets successfully.')

    # Create VAE network
    VAEmodel = MyVAE(z_dim).to(device)
    print('===> Initialized the network successfully.')

    # Create the agent
    agent = MyAgent(disp_interval=10)
    print('===> Initialized the agent successfully.')

    # Training loop
    if flag['resume_training']:
        print('===> Start training')
        for i in range(1, n_epochs + 1):
            agent.train(i)
            agent.test()
        print('End of training')

    # Plot
    if flag['plot_samples']:
        examples = enumerate(test_loader)
        batch_idx, example_data = next(examples)

        with torch.no_grad():
            output, _, _ = VAEmodel(example_data.to(device))
            output = output.cpu()
            fig = plt.figure(1)
            for i in range(6):
                plt.subplot(2,6,i+1)
                imshow_np(example_data[i,:])
                plt.subplot(2,6,i+1+6)
                imshow_np(output[i,:])

    if flag['plot_loss']:
        fig = plt.figure(2)
        train_losses, train_counter = agent.get_train_history()
        plt.plot(train_counter, train_losses, color='blue')
        # plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('Train loss')


    plt.show()
