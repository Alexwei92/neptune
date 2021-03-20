import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import genereate_sample_folder

class VAEGANTrain():
    """
    VAEGAN Training Agent
    """
    def __init__(self, model, device, is_eval, train_params, log_params):
        self.device = device
        self.model = model.to(self.device)
        self.z_dim = self.model.get_latent_dim()
        self.last_epoch = 0
        self.num_iter = 0
        self.epoch = []
        self.iteration = []
        self.netE_losses = []
        self.netG_losses = []
        self.netD_losses = []
        self.kld_losses = []
        self.kld_losses_dim_wise = []
        self.train_loader = None
        self.test_loader = None
        self.example_data = None

        # training parameters
        self.max_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.optimizerE = optim.Adam(self.model.netE.parameters(),
                                    lr=train_params['netE_learning_rate'],
                                    betas=(0.9, 0.999),
                                    weight_decay=train_params['weight_decay'])
        self.optimizerG = optim.Adam(self.model.netG.parameters(),
                                    lr=train_params['netG_learning_rate'],
                                    betas=(0.9, 0.999),
                                    weight_decay=train_params['weight_decay'])
        self.optimizerD = optim.Adam(self.model.netD.parameters(),
                                    lr=train_params['netD_learning_rate'],
                                    betas=(0.9, 0.999),
                                    weight_decay=train_params['weight_decay'])

        # logging parameters
        self.checkpoint_preload = log_params['checkpoint_preload']
        self.generate_samples = log_params['generate_samples']
        self.log_interval = log_params['log_interval']
        self.use_tensorboard = log_params['use_tensorboard']
        output_dir = log_params['output_dir']
        checkpoint_filename = log_params['checkpoint_filename']
        model_filename = log_params['model_filename']
        model_name = log_params['name']        

        checkpoint_filename = checkpoint_filename + '_z_' + str(self.z_dim) + '.pt'
        model_filename = model_filename + '_z_' + str(self.z_dim) + '.pt'
        self.checkpoint_filename = os.path.join(output_dir, checkpoint_filename)
        self.model_filename = os.path.join(output_dir, model_filename)
        self.sample_folder_path = os.path.join(output_dir, model_name + '_sample_folder_z_' + str(self.z_dim))

        # Use tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(output_dir, 'tb/' + model_name))

        # Preload a checkpoint
        if is_eval or self.checkpoint_preload:
            self.load_checkpoint(self.checkpoint_filename)

    def load_dataset(self, train_data, test_data):
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=6)
        # Generate sample folder
        if self.generate_samples:
            sample_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=6)
            self.example_data = genereate_sample_folder(self.sample_folder_path, sample_loader, self.checkpoint_preload)

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizerE.load_state_dict(checkpoint['optimizerE'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD'])
        self.epoch = checkpoint['epoch']
        self.last_epoch = self.epoch[-1]
        self.iteration = checkpoint['iteration']
        self.num_iter = self.iteration[-1]
        self.netE_losses = checkpoint['netE_losses']
        self.netG_losses = checkpoint['netG_losses']
        self.netD_losses = checkpoint['netD_losses']
        self.kld_losses = checkpoint['kld_losses']
        self.kld_losses_dim_wise = checkpoint['kld_losses_dim_wise']

    def train_batch(self, batch_x_real):
        batch_size = batch_x_real.size(0)
        y_real = torch.ones(batch_size).to(self.device)
        y_fake = torch.zeros(batch_size).to(self.device)

        # Extract fake images corresponding to real images
        mu, logvar = self.model.encode(batch_x_real)
        batch_z = self.model.reparameterize(mu, logvar)
        batch_x_fake = self.model.decode(batch_z.detach())

        # Extract fake images corresponding to noise (prior)
        batch_x_prior = self.model.sample(batch_size, self.device)

        # Compute D(x) for real and fake images along with their features
        l_real, _ = self.model.discriminate(batch_x_real)
        l_fake, _ = self.model.discriminate(batch_x_fake.detach())
        l_prior, _ = self.model.discriminate(batch_x_prior.detach())

        # 1) Update Discriminator
        loss_D = F.binary_cross_entropy(l_real, y_real) \
                + F.binary_cross_entropy(l_fake, y_fake)
                # + 0.5*F.binary_cross_entropy(l_prior, y_fake)
        self.optimizerD.zero_grad()
        loss_D.backward(retain_graph=True)
        self.optimizerD.step()

        # 2) Update Generator
        l_real, s_real = self.model.discriminate(batch_x_real)
        l_fake, s_fake = self.model.discriminate(batch_x_fake) 
        l_prior, s_prior = self.model.discriminate(batch_x_prior)
        loss_D = F.binary_cross_entropy(l_real, y_real) \
                + F.binary_cross_entropy(l_fake, y_fake)
                # + 0.5*F.binary_cross_entropy(l_prior, y_fake)

        feature_loss = F.mse_loss(s_fake, s_real, reduction='sum').div(batch_size)
        gamma = 1.0
        loss_G = gamma * feature_loss - loss_D
        self.optimizerG.zero_grad()
        loss_G.backward(retain_graph=True)
        self.optimizerG.step()
        
        # 3) Update Encoder
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_dim_wise = kld.mean(0)
        kld = kld.sum(1).mean(0)
        batch_x_fake = self.model.decode(batch_z)
        _, s_real = self.model.discriminate(batch_x_real)
        _, s_fake = self.model.discriminate(batch_x_fake)
        feature_loss = F.mse_loss(s_fake, s_real, reduction='sum').div(batch_size)

        beta = 3.0
        loss_E = beta * kld + feature_loss
        self.optimizerE.zero_grad()
        loss_E.backward()
        self.optimizerE.step()
    
        return {'loss_E': loss_E, 'loss_G': loss_G, 'loss_D': loss_D, 'KLD': kld, 'KLD_dim': kld_dim_wise}

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            netE_loss, netG_loss, netD_loss, kld_loss = 0.0, 0.0, 0.0, 0.0
            for batch_idx, batch_x, in enumerate(self.train_loader):
                self.num_iter += 1
                batch_x = batch_x.to(self.device)
                train_loss = self.train_batch(batch_x)

                netD_loss += train_loss['loss_D'].item()
                netG_loss += train_loss['loss_G'].item()
                netE_loss += train_loss['loss_E'].item()
                kld_loss += train_loss['KLD'].item()
                self.iteration.append(self.num_iter)
                self.netD_losses.append(train_loss['loss_D'].item())
                self.netG_losses.append(train_loss['loss_G'].item())
                self.netE_losses.append(train_loss['loss_E'].item())
                self.kld_losses.append(train_loss['KLD'].item())
                self.kld_losses_dim_wise.append(train_loss['KLD_dim'].data.cpu().numpy())

                if self.use_tensorboard:
                    self.writer.add_scalar('Train/netD_loss', train_loss['loss_D'].item(), self.num_iter)
                    self.writer.add_scalar('Train/netG_loss', train_loss['loss_G'].item(), self.num_iter)
                    self.writer.add_scalar('Train/netE_loss', train_loss['loss_E'].item(), self.num_iter)
                    self.writer.add_scalar('Train/kld_loss', train_loss['KLD'].item(), self.num_iter)

            # Test
            # test_losses = self.test()

            n_batch = len(self.train_loader)
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write("Epoch:{:d}, loss_E={:.4f}, loss_G={:.4f}, loss_D={:.4f}, KLD={:.4f}"
                .format(epoch + self.last_epoch, netE_loss/n_batch, netG_loss/n_batch, netD_loss/n_batch, kld_loss/n_batch))

            # Logging
            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                # self.save_model(self.model_filename)
                if self.generate_samples:
                    self.save_sample(self.get_current_epoch(), self.example_data, self.sample_folder_path)
        
        # End of training
        if self.use_tensorboard:
            self.writer.close()


    # def test(self):
    #     self.model.eval()
    #     test_loss = 0.0
    #     with torch.no_grad():
    #         for batch_idx, batch_x in enumerate(self.test_loader):
    #             batch_x = batch_x.to(self.device)

    #     return test_loss

    def save_sample(self, epoch, example_data, sample_folder_path):
        self.model.eval()
        with torch.no_grad():
            generated_sample, _, _ = self.model(example_data.to(self.device))
            vutils.save_image(generated_sample,
                              os.path.join(sample_folder_path, 'reconstructed_image_epoch_{:d}.png'.format(epoch)),
                              normalize=True,
                              range=(-1, 1))
            if self.use_tensorboard:
                img_grid = vutils.make_grid(generated_sample, normalize=True, range=(-1, 1))
                self.writer.add_image('reconstructed_image', img_grid, epoch)

    def save_checkpoint(self, file_path):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizerE': self.optimizerE.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'netE_losses': self.netE_losses,
            'netG_losses': self.netG_losses,
            'netD_losses': self.netD_losses,
            'kld_losses': self.kld_losses,
            'kld_losses_dim_wise': self.kld_losses_dim_wise,
        }, file_path)
        # print('Save checkpoint to ', file_path)

    def save_model(self):
        torch.save(self.model.state_dict(), file_path)
        # print('Save model to ', file_path)

    def get_current_epoch(self):
        return self.epoch[-1]