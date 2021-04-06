import os
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.utils as vutils
from tqdm import tqdm
import math

from imitation_learning import BaseTrain

class GANTrain(BaseTrain):
    '''
    GAN Training Agent
    '''
    def __init__(self,
                model,
                device,
                is_eval,
                train_params,
                log_params):
        
        super().__init__(model, device, is_eval, train_params, log_params)

    def configure(self, train_params, log_params):
        # optimizer
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
            'netG_loss': [],
            'netD_loss': [],
        }

        # checkpoint_dict
        self.checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'loss_history': self.loss_history,
            'tb_folder_name': self.tb_folder_name,
        }

    def train_batch(self, batch_real):
        batch_size = batch_real.size(0)
        label_real = torch.full((batch_size,), 0.9).to(self.device)
        label_fake = torch.full((batch_size,), 0.1).to(self.device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        output_real = self.model.discriminate(batch_real)
        errD_real = F.binary_cross_entropy(output_real, label_real)
        # train with fake
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        batch_fake = self.model.generate(noise)
        output_fake = self.model.discriminate(batch_fake.detach())
        errD_fake = F.binary_cross_entropy(output_fake, label_fake)
        errD = (errD_real + errD_fake) / 2.0
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        output = self.model.discriminate(batch_fake)
        errG = F.binary_cross_entropy(output, label_real)

        # Control training status
        train_netD = True
        train_netG = True
        if train_netG:
            self.model.netG.zero_grad()
            errG.backward(retain_graph=True)
            self.optimizerG.step()
        if train_netD:
            self.model.netD.zero_grad()
            errD.backward()
            self.optimizerD.step()

        return {'netD_loss': errD, 'netG_loss': errG}

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            netG_loss, netD_loss = 0.0, 0.0
            for batch_idx, batch_x in enumerate(self.train_dataloader):
                self.num_iter += 1
                batch_x = batch_x.to(self.device)
                train_loss = self.train_batch(batch_x)

                netD_loss += train_loss['netD_loss'].item()
                netG_loss += train_loss['netG_loss'].item()
                self.iteration.append(self.num_iter)
                for name in train_loss.keys():
                    self.loss_history[name].append(train_loss[name].item())

                if self.use_tensorboard:
                    for name in train_loss.keys():
                        self.writer.add_scalar('Train/' + name, train_loss[name].item(), self.num_iter)

            n_batch = len(self.train_dataloader)
            netD_loss /=  n_batch
            netG_loss /=  n_batch

            # Test
            test_netD_loss, test_netG_loss = self.test()

            # Logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, (train) loss_D = {:.3e}, loss_G = {:.3e} | (test) loss_D = {:.3e}, loss_G = {:.3e}'
                    .format(epoch + self.last_epoch,
                        netD_loss,
                        netG_loss,
                        test_netD_loss,
                        test_netG_loss)
                    )

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
        netD_loss, netG_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(self.test_dataloader):
                batch_x = batch_x.to(self.device)
                batch_size = batch_x.size(0)
                noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
                batch_fake = self.model.generate(noise)
                label_real = torch.ones(batch_size).to(self.device)
                label_fake = torch.zeros(batch_size).to(self.device)
                # Discriminator
                output1 = self.model.discriminate(batch_x)
                errD_real = F.binary_cross_entropy(output1, label_real)
                output2 = self.model.discriminate(batch_fake.detach())
                errD_fake = F.binary_cross_entropy(output2, label_fake)
                errD = (errD_real + errD_fake) / 2.0
                netD_loss += errD.item()
                # Generator
                errG = F.binary_cross_entropy(output2, label_real)
                netG_loss += errG.item()

        n_batch = len(self.test_dataloader)
        return netD_loss / n_batch, netG_loss / n_batch

    def save_sample(self, epoch, example_data, sample_folder_path):
        self.model.eval()
        with torch.no_grad():
            sample_img = self.model.sample(example_data.size(0), self.device)
            vutils.save_image(sample_img,
                              os.path.join(sample_folder_path, 'reconstructed_image_epoch_{:d}.png'.format(epoch)),
                              normalize=True,
                              range=(-1, 1))
            if self.use_tensorboard:
                img_grid = vutils.make_grid(sample_img, normalize=True, range=(-1, 1))
                self.writer.add_image('reconstructed_image', img_grid, epoch)
