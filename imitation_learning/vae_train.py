import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from utils import plot_generate_figure, plot_train_losses
from utils import genereate_sample_folder

class VAETrain():
    """
    VAE Training Agent
    """
    def __init__(self, model, device, is_eval, train_params, log_params):
        self.device = device
        self.model = model.to(self.device)
        self.z_dim = self.model.get_latent_dim()
        self.last_epoch = 0
        self.num_iter = 0
        self.epoch = []
        self.iteration = []
        self.total_losses = []
        self.mse_losses = []
        self.kld_losses = []
        self.kld_losses_z = []
        self.train_loader = None
        self.test_loader = None
        self.example_data = None

        # training parameters
        self.max_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=train_params['learning_rate'],
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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.last_epoch = self.epoch[-1]
        self.iteration = checkpoint['iteration']
        self.num_iter = self.iteration[-1]
        self.total_losses = checkpoint['total_losses']
        self.mse_losses = checkpoint['mse_losses']
        self.kld_losses = checkpoint['kld_losses']
        self.kld_losses_z = checkpoint['kld_losses_z']

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            total_losses = 0.0
            KLD_losses_dim_wise = np.zeros((self.model.z_dim,), dtype=np.float32)
            for batch_idx, batch_x in enumerate(self.train_loader):
                self.num_iter += 1
                batch_x = batch_x.to(self.device)
                batch_x_recon, mu, logvar = self.model(batch_x)
                train_loss = self.model.loss_function(batch_x_recon, batch_x, mu, logvar,
                                                    num_iter=self.num_iter)
                self.optimizer.zero_grad()
                train_loss['loss'].backward()
                self.optimizer.step()

                total_losses += train_loss['loss'].item()
                self.iteration.append(self.num_iter)
                self.total_losses.append(train_loss['loss'].item())
                self.mse_losses.append(train_loss['MSE'].item())
                self.kld_losses.append(train_loss['KLD'].item())
                self.kld_losses_z.append(train_loss['KLD_z'].data.cpu().numpy())

                if self.use_tensorboard:
                    self.writer.add_scalar('Train/total_loss', train_loss['loss'].item(), self.num_iter)
                    self.writer.add_scalar('Train/mse_loss', train_loss['MSE'].item(), self.num_iter)
                    self.writer.add_scalar('Train/kld_loss', train_loss['KLD'].item(), self.num_iter)

            # Test
            test_losses = self.test()

            n_batch = len(self.train_loader)
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, train loss = {:.3e} | test loss = {:.3e}'.
                        format(epoch + self.last_epoch, total_losses / n_batch, test_losses / n_batch))

            # Logging
            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)
                if self.generate_samples:
                    self.save_sample(self.get_current_epoch(), self.example_data, self.sample_folder_path)

        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def test(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(self.test_loader):
                batch_x = batch_x.to(self.device)
                batch_x_recon, mu, logvar = self.model(batch_x)
                train_loss = self.model.loss_function(batch_x_recon, batch_x, mu, logvar,
                                                    num_iter=self.num_iter)
                test_loss += train_loss['loss'].item()
        return test_loss

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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_losses': self.total_losses,
            'mse_losses': self.mse_losses,
            'kld_losses': self.kld_losses,
            'kld_losses_z': self.kld_losses_z,
        }, file_path)
        # print('Save checkpoint to ', file_path)

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        # print('Save model to ', file_path)

    def get_current_epoch(self):
        return self.epoch[-1]

    def get_train_history(self):
        return self.iteration, self.total_losses, self.mse_losses, self.kld_losses

    # def get_latent(self, x):
    #     z = self.model.get_latent(x)
    #     return z

    # def plot_generate_result(self, data_loader, N=6):
    #     examples = enumerate(data_loader)
    #     batch_idx, example_data = next(examples)
    #     with torch.no_grad():
    #         generated_data, _, _ = self.model(example_data.to(self.device))
    #         plot_generate_figure(generated_data.cpu(), example_data, N)
