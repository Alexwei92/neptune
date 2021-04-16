import os
import datetime
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class EndToEndTrain():
    """
    End to End Training Agent
    """
    def __init__(self,
                model,
                device,
                is_eval,
                train_params,
                log_params):
        
        # Model
        self.device = device
        self.model = model.to(device)

        # Dataloader
        self.train_dataloader = None
        self.test_dataloader = None
        self.example_data = None

        # Training parameters
        self.max_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
       
        # Logging parameters
        self.last_epoch = 0
        self.num_iter = 0
        self.epoch = []
        self.iteration = []
        self.tb_folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")

        self.checkpoint_preload = log_params['checkpoint_preload']
        self.log_interval = log_params['log_interval']
        self.use_tensorboard = log_params['use_tensorboard']
        self.model_name = log_params['name']
        log_folder = os.path.join(log_params['output_dir'], self.model_name)
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        checkpoint_filename = self.model_name + '_checkpoint.tar'
        model_filename = self.model_name + '_model.pt'
        self.checkpoint_filename = os.path.join(log_folder, checkpoint_filename)
        self.model_filename = os.path.join(log_folder, model_filename)

        # Optimizer and loss history configure
        self.configure(train_params, log_params)

        # Load a checkpoint
        if is_eval or self.checkpoint_preload:
            self.load_checkpoint(self.checkpoint_filename)

        # Use tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(log_folder, 'tb/' + self.tb_folder_name))

    def configure(self, train_params, log_params):
        # optimizer      
        self.optimizer = optim.Adam(self.model.parameters(),
                                lr=train_params['optimizer']['learning_rate'],
                                betas=eval(train_params['optimizer']['betas']),
                                weight_decay=train_params['optimizer']['weight_decay'])

        # loss history
        self.loss_history = {
            'total_loss': [],
        }

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
        self.loss_history = checkpoint['loss_history']
        self.tb_folder_name = checkpoint['tb_folder_name']

    def load_dataset(self, train_data, test_data):
        if train_data is not None:
            self.train_dataloader = DataLoader(train_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=6,
                                    drop_last=False)
        if test_data is not None:
            self.test_dataloader = DataLoader(test_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=6,
                                    drop_last=False)

    def train(self):
        # Start training
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            # Train
            self.model.train()
            train_total_loss = 0.0
            for batch_idx, batch_x in enumerate(self.train_dataloader):
                batch_img, batch_y = batch_x
                self.num_iter += 1
                batch_img = batch_img.to(self.device)
                batch_y_pred = self.model(batch_img).view(-1)
                train_loss = self.model.loss_function(batch_y_pred, batch_y.to(self.device))
                self.optimizer.zero_grad()
                train_loss['total_loss'].backward()
                self.optimizer.step()

                train_total_loss += train_loss['total_loss'].item()
                self.iteration.append(self.num_iter)
                for name in train_loss.keys():
                    self.loss_history[name].append(train_loss[name].item())

                if self.use_tensorboard:
                    for name in train_loss.keys():
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

        # End of training
        if self.use_tensorboard:
            self.writer.close()

    def test(self):
        self.model.eval()
        test_total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(self.test_dataloader):
                batch_img, batch_y = batch_x
                batch_img = batch_img.to(self.device)
                batch_y_pred = self.model(batch_img).view(-1)
                test_loss = self.model.loss_function(batch_y_pred, batch_y.to(self.device))
                test_total_loss += test_loss['total_loss'].item()
        
        n_batch = len(self.test_dataloader)
        return test_total_loss / n_batch
    
    def save_checkpoint(self, file_path):
        checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'tb_folder_name': self.tb_folder_name,
        }

        torch.save(checkpoint_dict, file_path)

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def get_current_epoch(self):
        return self.epoch[-1]

    def get_train_history(self):
        return self.iteration, self.loss_history