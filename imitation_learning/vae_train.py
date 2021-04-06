import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from imitation_learning import BaseTrain

class VAETrain(BaseTrain):
    """
    VAE Training Agent
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
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=train_params['optimizer']['learning_rate'],
                                betas=eval(train_params['optimizer']['betas']),
                                weight_decay=train_params['optimizer']['weight_decay'])

        # loss history
        self.loss_history = {
            'total_loss': [],
            'mse_loss': [],
            'kld_loss': [],
            'kld_loss_z': [],
        }

        # others
        if self.model_name == 'factor_vae':
            self.optimizerD = optim.Adam(self.model.parameters(),
                        lr=train_params['optimizerD']['learning_rate'],
                        betas=eval(train_params['optimizerD']['betas']),
                        weight_decay=train_params['optimizerD']['weight_decay'])
            
            self.D_z = None
            self.loss_history.update({
                'vae_tc_loss': [],
                'D_tc_loss': [],
            })

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            train_total_loss = 0.0
            for batch_idx, batch_x in enumerate(self.train_dataloader):
                if isinstance(batch_x, list):
                    [batch_x, batch_sample] = batch_x
                self.num_iter += 1
                batch_x = batch_x.to(self.device)
                results = self.model(batch_x)
                if self.model_name == 'factor_vae':
                    # VAE
                    train_loss_vae = self.model.loss_function(*results, mode='VAE')
                    self.optimizer.zero_grad()
                    train_loss_vae['total_loss'].backward(retain_graph=True)
                    self.optimizer.step()
                    # Discriminator
                    batch_sample = batch_sample.to(self.device)
                    results_sample = self.model(batch_sample)
                    train_loss_D = self.model.loss_function(*results_sample, mode='netD')
                    self.optimizerD.zero_grad()
                    train_loss_D['D_tc_loss'].backward()
                    self.optimizerD.step()
                    train_loss = {**train_loss_vae, **train_loss_D}
                else:
                    train_loss = self.model.loss_function(*results, num_iter=self.num_iter)
                    self.optimizer.zero_grad()
                    train_loss['total_loss'].backward()
                    self.optimizer.step()

                train_total_loss += train_loss['total_loss'].item()
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
            train_total_loss /= n_batch
            
            # Test
            # test_total_loss = self.test()

            # Logging
            self.epoch.append(epoch + self.last_epoch)
            tqdm.write('Epoch: {:d}, train loss = {:.3e}'.
                        format(epoch + self.last_epoch,
                            train_total_loss))
            # tqdm.write('Epoch: {:d}, train loss = {:.3e} | test loss = {:.3e}'.
            #             format(epoch + self.last_epoch,
            #                 train_total_loss,
            #                 test_total_loss))
            
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
        test_total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(self.test_dataloader):
                if isinstance(batch_x, list):
                    [batch_x, batch_sample] = batch_x
                batch_x = batch_x.to(self.device)
                results = self.model(batch_x)
                test_loss = self.model.loss_function(*results, num_iter=self.num_iter, mode='VAE')
                test_total_loss += test_loss['total_loss'].item()
        
        n_batch = len(self.test_dataloader)
        return test_total_loss / n_batch

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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'tb_folder_name': self.tb_folder_name,
        }

        if self.model_name == 'factor_vae':
            checkpoint_dict.update({
                'optimizerD_state_dict': self.optimizerD.state_dict(),
            })

        torch.save(checkpoint_dict, file_path)