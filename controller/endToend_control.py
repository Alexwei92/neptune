import torch
import os
import cv2
import numpy as np
from torchvision import transforms
import yaml

from models import *
from utils import *

class EndToEndCtrl():
    '''
    End To End Controller
    '''
    def __init__(self,
                model_path,
                img_resize): 

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)
        self.resize = img_resize

        print('The latent controller is initialized.')

    def load_model(self, model_path):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        file = open(os.path.join(parent_dir, 'configs', 'end_to_end.yaml'), 'r')
        model_config = yaml.safe_load(file)
        file.close()
        
        self.in_channels = model_config['model_params']['in_channels']
        if self.in_channels == 1: # Depth
            self.transform_composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)), # from [0,255] to [-1,1]
            ])
        elif self.in_channels == 3: # RGB
            self.transform_composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # from [0,255] to [-1,1]
            ])
        elif self.in_channels == 4: # RGB-D
            self.transform_composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)), # from [0,255] to [-1,1]
            ])  

        model = torch.load(model_path)
        self.model = EndToEnd(**model_config['model_params']).to(self.device)
        self.model.load_state_dict(model)

    def predict(self, image_color, image_depth):

        if self.in_channels == 1:
            image_np = image_depth.copy()
            image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
            image_np = np.reshape(image_np, (self.resize[1], self.resize[0], 1))
        elif self.in_channels == 3:
            image_np = image_color.copy()
            image_np = cv2.resize(image_np, (self.resize[1], self.resize[0]))
        elif self.in_channels == 4:
            image_color = image_color.copy()
            image_depth = image_depth.copy()
            image_color = cv2.resize(image_color, (self.resize[1], self.resize[0]))
            image_depth = cv2.resize(image_depth, (self.resize[1], self.resize[0]))
            image_depth = np.reshape(image_depth, (self.resize[1], self.resize[0], 1))
            image_np = np.concatenate((image_color, image_depth), axis=2)
        image_tensor = self.transform_composed(image_np)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(image_tensor.unsqueeze(0).to(self.device))
            y_pred = y_pred.cpu().item()
        
        if np.abs(y_pred) < 1e-2:
            y_pred = 0.0
        
        return np.clip(y_pred, -1.0, 1.0)