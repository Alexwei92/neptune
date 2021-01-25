import setup_path
import glob
import os

from imitation_learning.regression_train import LGTrain
from imitation_learning.vae_train import VAETrain
from imitation_learning.nn_train import NNTrain

import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from sklearn.model_selection import train_test_split

from utils import *
from models import *

if __name__ == '__main__':

    ########## User-defined Parameters ##########
    dataset_dir = os.path.join(setup_path.parent_dir, 'my_datasets/nh') # dataset folder path
    output_dir = os.path.join(setup_path.parent_dir, 'my_outputs/nh/iter0') # output folder path
    #############################################

    if not os.path.isdir(dataset_dir):
        raise IOError("***No such folder!", dataset_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('==============================================')

    # # 1) Linear Regression
    # ########## User-defined Parameters ##########
    # weight_file_name = 'ling_weight_result.csv'
    # #############################################
    # # Create the agent
    # ling_agent = LGTrain(dataset_dir, preload=True)
    # print('===> Initialized LGTrain agent successfully.')

    # # Train
    # ling_agent.train()
    # print('===> Trained linear regression model successfully!')

    # # Save weight to file
    # ling_agent.save_weight(os.path.join(output_dir, weight_file_name))
    # print('==============================================')

    # 2) VAE
    ########## User-defined Parameters ##########
    batch_size = 32 # batch size
    n_epochs = 200 # number of epochs
    z_dim = 10 # latent variable dimension
    checkpoint_file_name = 'VAE_checkpoint_z_'+ str(z_dim) + '.pt'
    # torch.backends.cudnn.deterministic = True
    #############################################
    # DataLoader
    # all_data = ImageDataset(dataset_dir, resize=(64, 64))
    # train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=11) # split into train and test datasets

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=6)
    # print('===> Loaded VAE datasets successfully.')

    # Create the agent
    vae_agent = VAETrain(MyVAE(z_dim))
    vae_agent.load_checkpoint(os.path.join(output_dir, checkpoint_file_name))
    print('===> Initialized VAETrain agent successfully.')

    # # Training loop
    # for epoch in range(1, n_epochs + 1):
    #     vae_agent.train(epoch, train_loader)
    #     vae_agent.test(test_loader)
    #     # Save to checkpoint
    #     if epoch % 10 == 0 and epoch > 0:
    #         vae_agent.save_checkpoint(epoch, os.path.join(output_dir, checkpoint_file_name))
    # print('===> Trained VAE model successfully.')    

    # # 3) Plot VAE results
    # # Plot generated figures
    # examples = enumerate(test_loader)
    # batch_idx, example_data = next(examples)
    # with torch.no_grad():
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     generated_data, _, _ = vae_agent.VAEmodel(example_data.to(device))
    #     plot_generate_figure(generated_data.cpu(), example_data, disp_N=2)
        
    # # Plot training history
    # plot_train_losses(vae_agent.get_train_history())
    # plt.show()

    # 4) Controller Network
    ########## User-defined Parameters ##########
    batch_size = 32 # batch size
    n_epochs = 400 # number of epochs
    z_dim = 10 # latent variable dimension
    checkpoint_file_name = 'NN_checkpoint_z_'+ str(z_dim) + '.pt'
    # torch.backends.cudnn.deterministic = True
    #############################################
    data = LatentDataset(dataset_dir, resize=(64, 64))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)

    nn_agent = NNTrain(MyNN(z_dim))
    # nn_agent.load_checkpoint(os.path.join(output_dir, checkpoint_file_name))
    print('===> Initialized NNTrain agent successfully.')

    # Training loop
    for epoch in range(1, n_epochs + 1):
        nn_agent.train(epoch, data_loader, vae_agent)
        # nn_agent.test(test_loader)
        # Save to checkpoint
        if epoch % 10 == 0 and epoch > 0:
            nn_agent.save_checkpoint(epoch, os.path.join(output_dir, checkpoint_file_name))
    print('===> Trained VAE model successfully.')   

    # Plot training history
    plot_train_losses2(nn_agent.get_train_history())
    plt.show()