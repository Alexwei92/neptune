import torch
import os
import cv2
import numpy as np

from controller import BaseCtrl
from models import *
from utils import *

class NNCtrl(BaseCtrl):
    '''
    Neural Network Controller
    '''
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_z = kwargs.get('n_z')
        self.load_VAE_model(kwargs.get('vae_model_path'), n_z)
        self.load_NN_model(kwargs.get('nn_model_path'), n_z)

    def load_VAE_model(self, file_path, n_z):
        checkpoint = torch.load(file_path)
        self.VAEmodel = MyVAE(n_z)
        self.VAEmodel.load_state_dict(checkpoint['model_state_dict'])

    def load_NN_model(self, file_path, n_z):
        checkpoint = torch.load(file_path)
        self.NNmodel = MyNN(n_z)
        self.NNmodel.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, image_color):
        image = cv2.resize(image_color, (64,64))
        image = normalize_image(image)
        image = torch.from_numpy(image)

        self.VAEmodel.eval()
        self.NNmodel.eval()
        with torch.no_grad():
            z = self.VAEmodel.get_latent(image.to(self.device))
            y = self.NNmodel(z)
            y = y.cpu().item()

        # TODO: Normalize and restrict the output
        if y > 1.0:
            y = 1.0
        if y < -1.0:
            y = -1.0
        return y
        
    def step(self, yaw_cmd, flight_mode):
        if flight_mode == 'hover':
            self.send_command(0.0, is_hover=True)
        elif flight_mode == 'mission':
            self.send_command(yaw_cmd, is_hover=False)
        else:
            print('Unknown flight_mode: ', flight_mode)
            raise Exception