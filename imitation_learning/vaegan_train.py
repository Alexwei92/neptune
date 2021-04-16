import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from imitation_learning import BaseTrain

class VAEGANTrain(BaseTrain):
    """
    VAEGAN Training Agent
    """
    def __init__(self,
                model,
                device,
                is_eval,
                train_params,
                log_params):
        
        super().__init__(model, device, is_eval, train_params, log_params)

    def configure(self, train_params, log_params):
        # optimizer
        self.optimizerE = optim.Adam(self.model.netE.parameters(),
                                lr=train_params['optimizerE']['learning_rate'],
                                betas=eval(train_params['optimizerE']['betas']),
                                weight_decay=train_params['optimizerE']['weight_decay'])
        self.optimizerG = optim.Adam(self.model.netG.parameters(),
                                lr=train_params['optimizerG']['learning_rate'],
                                betas=eval(train_params['optimizerG']['betas']),
                                weight_decay=train_params['optimizerG']['weight_decay'])
        self.optimizerD = optim.Adam(self.model.netD.parameters(),
                                lr=train_params['optimizerD']['learning_rate'],
                                betas=eval(train_params['optimizerD']['betas']),
                                weight_decay=train_params['optimizerD']['weight_decay'])

        # loss history
        self.loss_history = {
            'netE_loss': [],
            'netG_loss': [],
            'netD_loss': [],
            'kld_loss': [],
            'kld_loss_z': [],
            'mse_loss': [],
            'feature_loss': [],
        }

        # beta-VAE
        self.C_max = torch.Tensor([13]).to(self.device)
        self.C_stop_iter = 1e4
        self.gamma = 10
        self.beta = 1

    def train_batch(self, batch_x_real):
        batch_size = batch_x_real.size(0)
        y_real = torch.full((batch_size,), 0.9).to(self.device)
        y_fake = torch.full((batch_size,), 0.1).to(self.device)
        # Extract fake images corresponding to real images
        mu, logvar = self.model.encode(batch_x_real)
        batch_z = self.model.reparameterize(mu, logvar)
        batch_x_fake = self.model.decode(batch_z.detach())
        # Extract fake images corresponding to noise (prior)
        batch_x_prior = self.model.sample(batch_size, self.device)
        #######################
        # (1) Update Discriminator
        #######################
        l_real, _ = self.model.discriminate(batch_x_real)
        l_fake, _ = self.model.discriminate(batch_x_fake.detach())
        l_prior, _ = self.model.discriminate(batch_x_prior.detach())
        loss_D = F.binary_cross_entropy(l_real, y_real) \
                + F.binary_cross_entropy(l_fake, y_fake) \
                + F.binary_cross_entropy(l_prior, y_fake)
        self.optimizerD.zero_grad()
        loss_D.backward(retain_graph=True)
        self.optimizerD.step()
        ######################
        # (2) Update Generator
        ######################
        l_real, s_real = self.model.discriminate(batch_x_real)
        l_fake, s_fake = self.model.discriminate(batch_x_fake) 
        l_prior, s_prior = self.model.discriminate(batch_x_prior)
        loss_D = F.binary_cross_entropy(l_real, y_real) \
                + F.binary_cross_entropy(l_fake, y_fake) \
                + F.binary_cross_entropy(l_prior, y_fake)
        feature_loss = F.mse_loss(s_fake, s_real, reduction='sum').div(batch_size)
        gamma = 1e-3
        loss_G = gamma * feature_loss - loss_D
        self.optimizerG.zero_grad()
        loss_G.backward(retain_graph=True)
        self.optimizerG.step()
        #####################
        # (3) Update Encoder
        #####################  
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # kld_z = kld.mean(0)
        kld = kld.sum(1).mean(0)
        batch_x_fake = self.model.decode(batch_z)
        _, s_fake = self.model.discriminate(batch_x_fake)
        feature_loss = F.mse_loss(s_fake, s_real, reduction='sum').div(batch_size)
        # beta-VAE
        # C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        # loss_E = feature_loss + self.gamma * (kld - C).abs()
        loss_E = feature_loss + self.beta * kld
        self.optimizerE.zero_grad()
        loss_E.backward()
        self.optimizerE.step()
        #######################
        # reconstrunction loss
        #######################
        mse_loss = F.mse_loss(batch_x_real, batch_x_fake, reduction='sum').div(batch_size)

        # Control training status
        train_netD = True
        train_netG = True
        train_netE = True

        return {'netE_loss': loss_E,
                'netG_loss': loss_G,
                'netD_loss': loss_D,
                'kld_loss': kld,
                # 'kld_loss_z': kld_z,
                'mse_loss': mse_loss,
                'feature_loss': feature_loss}

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            netE_loss, netG_loss, netD_loss, kld_loss = 0.0, 0.0, 0.0, 0.0
            for batch_idx, batch_x, in enumerate(self.train_dataloader):
                if isinstance(batch_x, list):
                    [batch_x, batch_sample] = batch_x
                self.num_iter += 1
                batch_x = batch_x.to(self.device)
                train_loss = self.train_batch(batch_x)

                netD_loss += train_loss['netD_loss'].item()
                netG_loss += train_loss['netG_loss'].item()
                netE_loss += train_loss['netE_loss'].item()
                kld_loss += train_loss['kld_loss'].item()
                self.iteration.append(self.num_iter)
                for name in train_loss.keys():
                    if name == 'kld_loss_z':
                        self.loss_history[name].append(train_loss[name].data.cpu().numpy())
                    else:
                        self.loss_history[name].append(train_loss[name].item())

                if self.use_tensorboard:
                    for name in train_loss.keys():
                        if name != 'kld_loss_z':
                            self.writer.add_scalar('Train/' + name, train_loss[name].item(), self.num_iter)

            n_batch = len(self.train_dataloader)
            netE_loss /= n_batch
            netG_loss /= n_batch
            netD_loss /= n_batch
            kld_loss /= n_batch

            # Test
            self.test()

            # Logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write("Epoch:{:d}, loss_E={:.4f}, loss_G={:.4f}, loss_D={:.4f}, KLD={:.4f}"
                .format(epoch + self.last_epoch,
                    netE_loss,
                    netG_loss,
                    netD_loss,
                    kld_loss))

            if epoch % self.log_interval == 0:
                self.save_checkpoint(self.checkpoint_filename)
                self.save_model(self.model_filename)
                if self.generate_samples:
                    self.save_sample(self.get_current_epoch(), self.example_data, self.sample_folder_path)
        
        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def test(self):
        pass

    def latent_traversal(self, img, traversal):
        img_all = torch.empty((1,3,64,64), requires_grad=False).to(self.device)
        if isinstance(img, tuple):
            img = img[0]

        self.model.eval()
        with torch.no_grad():
            z_raw_gpu = self.model.get_latent(img.unsqueeze(0).to(self.device))
            z_raw_cpu = z_raw_gpu.cpu().numpy()
            for i in range(self.z_dim):
                z_new_cpu = z_raw_cpu.copy()
                for value in traversal:
                    z_new_cpu[0, i] = value
                    z_new_gpu = torch.from_numpy(z_new_cpu.astype(np.float32)).unsqueeze(0).to(self.device)
                    img_new_gpu = self.model.decode(z_new_gpu)
                    img_all = torch.cat((img_all, img_new_gpu), axis=0)
        
        return img_all

    def save_checkpoint(self, file_path):
        checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizerE_state_dict': self.optimizerE.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'loss_history': self.loss_history,
            'tb_folder_name': self.tb_folder_name,
        }

        torch.save(checkpoint_dict, file_path)