import torch
import os
import cv2
import numpy as np
from torchvision import transforms
import yaml

from models import *
from utils import *

class LatentCtrl():
    '''
    Latent Controller
    '''
    def __init__(self,
                vae_model_path,
                vae_model_type,
                latent_model_path,
                latent_model_type,
                img_resize): 

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_VAE_model(vae_model_path, vae_model_type)
        self.load_latent_model(latent_model_path, latent_model_type)
        self.resize = img_resize

        self.transform_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
        ])
        print('The latent controller is initialized.')

    def load_VAE_model(self, model_path, model_type):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        file = open(os.path.join(parent_dir, 'configs', model_type + '.yaml'), 'r')
        model_config = yaml.safe_load(file)
        file.close()

        model = torch.load(model_path)
        self.VAE_model = vae_model[model_type](**model_config['model_params']).to(self.device)
        self.VAE_model.load_state_dict(model)

    def load_latent_model(self, model_path, model_type):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        file = open(os.path.join(parent_dir, 'configs', model_type + '.yaml'), 'r')
        model_config = yaml.safe_load(file)
        file.close()

        num_prvs = model_config['model_params']['num_prvs']
        prvs_mode = model_config['model_params']['prvs_mode']
        self.num_prvs = num_prvs
        if num_prvs > 0:
            if prvs_mode == 'exponential':
                self.prvs_index = exponential_decay(num_prvs)
            elif prvs_mode == 'linear':
                self.prvs_index = [i for i in reversed(range(1, num_prvs+1))]

        model = torch.load(model_path)
        self.Latent_model = latent_model[model_type](**model_config['model_params']).to(self.device)
        self.Latent_model.load_state_dict(model)

    def predict(self, image_color, yawRate, cmd_history):
        image_np = image_color.copy()
        image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
        image_tensor = self.transform_composed(image_np)
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0

        self.VAE_model.eval()
        self.Latent_model.eval()
        with torch.no_grad():
            z = self.VAE_model.get_latent(image_tensor.unsqueeze(0).to(self.device))
            state_extra = []
            if self.num_prvs > 0:
                for index in self.prvs_index:
                    state_extra = np.append(state_extra, cmd_history[-index])
            state_extra = np.append(state_extra, yawRate_norm)
            state_extra = state_extra.astype(np.float32)
            data = torch.cat([z, torch.from_numpy(state_extra).unsqueeze(0).to(self.device)], axis=1)
            y_pred = self.Latent_model(data)
            y_pred = y_pred.cpu().item()
        
        if np.abs(y_pred) < 1e-3:
            y_pred = 0.0
        
        return np.clip(y_pred, -1.0, 1.0)
    