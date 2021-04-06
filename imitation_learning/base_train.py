import os
import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import genereate_sample_folder

class BaseTrain():
    """
    Base Training Agent
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
        self.z_dim = self.model.get_latent_dim()

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
        self.generate_samples = log_params['generate_samples']
        self.log_interval = log_params['log_interval']
        self.use_tensorboard = log_params['use_tensorboard']
        self.model_name = log_params['name']
        log_folder = os.path.join(log_params['output_dir'], self.model_name)
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        checkpoint_filename = self.model_name + '_checkpoint_z_' + str(self.z_dim) + '.tar'
        model_filename = self.model_name + '_model_z_' + str(self.z_dim) + '.pt'
        self.checkpoint_filename = os.path.join(log_folder, checkpoint_filename)
        self.model_filename = os.path.join(log_folder, model_filename)
        self.sample_folder_path = os.path.join(log_folder, self.model_name + '_sample_folder_z_' + str(self.z_dim))

        # Optimizer and loss history configure
        self.configure(train_params, log_params)

        # Load a checkpoint
        if is_eval or self.checkpoint_preload:
            self.load_checkpoint(self.checkpoint_filename)

        # Use tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(log_folder, 'tb/' + self.tb_folder_name))

    def load_checkpoint(self, file_path):
        if not os.path.isfile(file_path):
            raise IOError("***No such file!", file_path)

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'optimizerD_state_dict' in checkpoint:
            self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        if 'optimizerG_state_dict' in checkpoint:
            self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        if 'optimizerE_state_dict' in checkpoint:
            self.optimizerE.load_state_dict(checkpoint['optimizerE_state_dict'])
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
            # Generate sample folder
            if self.generate_samples:
                sample_dataloader = DataLoader(test_data,
                                    batch_size=64,
                                    shuffle=True,
                                    num_workers=6,
                                    drop_last=True)
                self.example_data = genereate_sample_folder(self.sample_folder_path, sample_dataloader, self.checkpoint_preload)       

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def save_sample(self, epoch, example_data, sample_folder_path):
        self.model.eval()
        with torch.no_grad():
            results = self.model(example_data.to(self.device))
            vutils.save_image(results[0],
                              os.path.join(sample_folder_path, 'reconstructed_image_epoch_{:d}.png'.format(epoch)),
                              normalize=True,
                              range=(-1, 1))
            if self.use_tensorboard:
                img_grid = vutils.make_grid(results[0], normalize=True, range=(-1, 1))
                self.writer.add_image('reconstructed_image', img_grid, epoch)

    def get_current_epoch(self):
        return self.epoch[-1]

    def get_train_history(self):
        return self.iteration, self.loss_history

    def configure(self, train_params, log_params):
        # optimizer
        ##
        # loss history
        ## 
        # others
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
 
    def test(self):
        raise NotImplementedError

    def save_checkpoint(self, file_path):
        raise NotImplementedError
