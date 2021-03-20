import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import genereate_sample_folder

class GANTrain():
    '''
    GAN Training Agent
    '''
    def __init__(self, model, device, is_eval, train_params, log_params):
        self.device = device
        self.model = model.to(self.device)
        self.z_dim = self.model.get_latent_dim()
        self.last_epoch = 0
        self.num_iter = 0
        self.epoch = []
        self.iteration = []
        self.netG_losses = []
        self.netD_losses = []
        self.train_loader = None
        self.test_loader = None
        self.example_data = None

        # training parameters
        self.max_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
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
            self.example_data = torch.randn(64, self.z_dim, 1, 1).to(self.device)

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.epoch = checkpoint['epoch']
        self.last_epoch = self.epoch[-1]
        self.iteration = checkpoint['iteration']
        self.num_iter = self.iteration[-1]
        self.netG_losses = checkpoint['netG_losses']
        self.netD_losses = checkpoint['netD_losses']

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            netG_loss, netD_loss = 0.0, 0.0
            for batch_idx, batch_x in enumerate(self.train_loader):
                self.num_iter += 1
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.model.netD.zero_grad()
                batch_real = batch_x.to(self.device)
                batch_size = batch_real.size(0)
                label_real = torch.ones(batch_size).to(self.device)
                label_fake = torch.zeros(batch_size).to(self.device)

                output = self.model.discriminate(batch_real)
                errD_real = F.binary_cross_entropy(output, label_real)
                errD_real.backward()
                # D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
                batch_fake = self.model.generate(noise)
                output = self.model.discriminate(batch_fake.detach())
                errD_fake = F.binary_cross_entropy(output, label_fake)
                errD_fake.backward()
                # D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                netD_loss += errD.item()
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.model.netG.zero_grad()
                output = self.model.discriminate(batch_fake)
                errG = F.binary_cross_entropy(output, label_real)
                errG.backward()
                # D_G_z2 = output.mean().item()
                netG_loss += errG.item()
                self.optimizerG.step()
            
                self.iteration.append(self.num_iter)
                self.netD_losses.append(errD.item())
                self.netG_losses.append(errG.item())

                if self.use_tensorboard:
                    self.writer.add_scalar('Train/netD_loss', errD.item(), self.num_iter)
                    self.writer.add_scalar('Train/netG_loss', errG.item(), self.num_iter)

            # Test


            n_batch = len(self.train_loader)
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, loss_D = {:.4f}, loss_G = {:.4f}'
                .format(epoch + self.last_epoch, netD_loss/n_batch, netG_loss/n_batch))

            # Logging
            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)
                if self.generate_samples:
                    self.save_sample(self.get_current_epoch(), self.example_data, self.sample_folder_path) 

        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def save_sample(self, epoch, example_data, sample_folder_path):
        self.model.eval()
        with torch.no_grad():
            generated_sample = self.model.generate(example_data.to(self.device))
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
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'netG_losses': self.netG_losses,
            'netD_losses': self.netD_losses,
        }, file_path)
        # print('Save checkpoint to ', file_path)

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        # print('Save model to ', file_path)

    def get_current_epoch(self):
        return self.epoch[-1]

    def get_train_history(self):
        return self.iteration, self.netG_losses, self.netD_losses
