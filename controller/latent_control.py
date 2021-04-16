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
                img_resize): 

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_VAE_model(vae_model_path, vae_model_type)
        self.load_latent_model(latent_model_path)
        self.resize = img_resize

        self.transform_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # from [0,255] to [-1,1]
        ])
        print('The latent controller is initialized.')

    def load_VAE_model(self, model_path, model_type):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        file = open(os.path.join(parent_dir, 'configs', model_type + '.yaml'), 'r')
        model_config = yaml.safe_load(file)
        file.close()

        model = torch.load(model_path)
        if model_type in vae_model:
            self.VAE_model = vae_model[model_type](**model_config['model_params']).to(self.device)
        elif model_type in vaegan_model:
            self.VAE_model = vaegan_model[model_type](**model_config['model_params']).to(self.device)
        self.VAE_model.load_state_dict(model)
        self.z_dim = self.VAE_model.z_dim

    def load_latent_model(self, model_path):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        file = open(os.path.join(parent_dir, 'configs', 'latent_nn.yaml'), 'r')
        model_config = yaml.safe_load(file)
        file.close()

        self.with_yawRate = model_config['model_params']['with_yawRate']
        model = torch.load(model_path)
        model_config['model_params']['z_dim'] = self.z_dim
        self.Latent_model = LatentNN(**model_config['model_params']).to(self.device)
        self.Latent_model.load_state_dict(model)

    def predict(self, image_color, yawRate, cmd_history):
        image_np = image_color.copy()
        image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
        image_tensor = self.transform_composed(image_np)
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0
        
        self.VAE_model.eval()
        self.Latent_model.eval()
        with torch.no_grad():
            z = self.VAE_model.get_latent(image_tensor.unsqueeze(0).to(self.device), with_logvar=False)
            state_extra = []
            if self.with_yawRate:
                state_extra = np.append(state_extra, yawRate_norm)
            if len(state_extra) > 0:
                state_extra = state_extra.astype(np.float32)
                x = torch.cat((z, torch.from_numpy(state_extra).unsqueeze(0).to(self.device)), axis=1)
                y_pred = self.Latent_model(x)
            else:
                y_pred = self.Latent_model(z)
                
            y_pred = y_pred.cpu().item()
        

        # alpha = 0.5
        # y_pred = alpha * y_pred + (1 - alpha) * cmd_history[-1]
        if np.abs(y_pred) < 1e-2:
            y_pred = 0.0
        
        return np.clip(y_pred, -1.0, 1.0)
    