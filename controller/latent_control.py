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
    def __init__(self, vae_model_path, latent_model_path, z_dim, img_resize): 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_VAE_model(vae_model_path, z_dim)
        self.load_latent_model(latent_model_path, z_dim)
        self.resize = img_resize
        print('The latent controller is initialized.')

    def load_VAE_model(self, file_path, z_dim):
        model = torch.load(file_path)
        self.VAE_model = MyVAE(z_dim).to(self.device)
        self.VAE_model.load_state_dict(model['model_state_dict'])

    def load_latent_model(self, file_path, z_dim):
        model = torch.load(file_path)
        self.latent_model = MyLatent(z_dim).to(self.device)
        self.latent_model.load_state_dict(model['model_state_dict'])

    def predict(self, image_color):
        image_np = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
        image_norm = normalize_image(image_np)
        image_norm = image_norm.transpose((2,1,0)).astype(np.float32)
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0)

        # print(image.transpose((2,1,0)).shape)
        self.VAE_model.eval()
        self.latent_model.eval()
        with torch.no_grad():
            z = self.VAE_model.get_latent(image_tensor.to(self.device))
            # print(z.shape)
            y = self.latent_model(z)
            y = y.cpu().item()

        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0
        return y