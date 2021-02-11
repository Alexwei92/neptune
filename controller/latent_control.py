import torch
import os
import cv2
import numpy as np

from models import *
from utils import *

class LatentCtrl():
    '''
    Latent Controller
    '''
    def __init__(self, vae_model_path, latent_model_path, z_dim, num_prvs, img_resize): 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_VAE_model(vae_model_path, z_dim)
        self.load_latent_model(latent_model_path, z_dim, num_prvs)
        self.resize = img_resize
        self.num_prvs = num_prvs
        self.prvs_index = [5,4,3,2,1]
        self.cmd_prvs = np.zeros(max(self.prvs_index), dtype=np.float32)
        print('The latent controller is initialized.')

    def load_VAE_model(self, file_path, z_dim):
        model = torch.load(file_path)
        self.VAE_model = MyVAE(z_dim).to(self.device)
        self.VAE_model.load_state_dict(model['model_state_dict'])

    def load_latent_model(self, file_path, z_dim, num_prvs):
        model = torch.load(file_path)
        self.latent_model = MyLatent(z_dim+num_prvs+1).to(self.device)
        self.latent_model.load_state_dict(model['model_state_dict'])

    def predict(self, image_color, yawRate):
        image_np = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
        image_norm = normalize_image(image_np)
        image_norm = image_norm.transpose((2,1,0)).astype(np.float32)
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0)

        self.VAE_model.eval()
        self.latent_model.eval()
        with torch.no_grad():
            z = self.VAE_model.get_latent(image_tensor.to(self.device))
            state_extra = yawRate
            for index in self.prvs_index:
                state_extra = np.append(state_extra, self.cmd_prvs[index-1])
            state_extra = state_extra.astype(np.float32)
            data = torch.cat([z, torch.from_numpy(state_extra).unsqueeze(0).to(self.device)], axis=1)
            y = self.latent_model(data)
            y = y.cpu().item()
        
        # if np.abs(y) < 1e-3:
        #     y = 0.0
        
        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0

        for i in reversed(range(1,len(self.cmd_prvs))):
            self.cmd_prvs[i] = self.cmd_prvs[i-1]
        self.cmd_prvs[0] = y

        return y
    
    def reset_prvs(self):
        self.cmd_prvs.fill(0.0)