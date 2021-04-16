import setup_path
import numpy as np
import pandas
import torch
import yaml
import os
import glob
import cv2
import time
from torchvision import transforms
import math
import matplotlib.pyplot as plt

from utils import plot_with_cmd_compare
from models import *
from imitation_learning import exponential_decay

def read_data(folder_path, num_prvs=5, prvs_mode='linear'):
    # visual feature
    feature_path = os.path.join(folder_path, 'feature_preload.pkl')
    X = pandas.read_pickle(feature_path).to_numpy()

    # telemetry feature
    telemetry_data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
    timestamp = telemetry_data['timestamp'][:-1].to_numpy(float)
    timestamp -= timestamp[0]
    timestamp *= 1e-9

    # yaw cmd
    yaw_cmd = telemetry_data['yaw_cmd'][:-1].to_numpy()

    # yaw rate
    yawRate = np.reshape(telemetry_data['yaw_rate'][:-1].to_numpy(), (-1,1))

    # previous commands with time decaying
    if num_prvs > 0:
        y_prvs = np.zeros((len(yaw_cmd), num_prvs))
        if prvs_mode == 'exponential':
            prvs_index = exponential_decay(num_prvs)
        elif prvs_mode == 'linear':
            prvs_index = [i for i in reversed(range(1, num_prvs+1))]
        for i in range(len(yaw_cmd)):
            for j in range(num_prvs):
                y_prvs[i,j] = yaw_cmd[max(i-prvs_index[j], 0)] # [t-N,...t-1]
        X_extra = np.concatenate((y_prvs, yawRate), axis=1)
    else:
        X_extra = yawRate
    
    # total feature
    X = np.column_stack((X, X_extra))

    # image
    file_list_color = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    file_list_color.sort()

    return timestamp, yaw_cmd, X, file_list_color

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
            z = self.VAE_model.get_latent(image_tensor.unsqueeze(0).to(self.device), with_logvar=True)
            result = self.VAE_model(image_tensor.unsqueeze(0).to(self.device))
            state_extra = []
            if self.num_prvs > 0:
                for index in self.prvs_index:
                    state_extra = np.append(state_extra, cmd_history[-index])
            state_extra = np.append(state_extra, yawRate_norm)
            state_extra = state_extra.astype(np.float32)
            data = torch.cat([z, torch.from_numpy(state_extra).unsqueeze(0).to(self.device)], axis=1)
            y_pred = self.Latent_model(data)
            y_pred = y_pred.cpu().item()
        
        if np.abs(y_pred) < 1e-2:
            y_pred = 0.0
        
        return np.clip(y_pred, -1.0, 1.0), z.cpu().numpy()[0], result[0].cpu().squeeze(0), result[1].cpu().squeeze(0)

if __name__ == '__main__':
    folder_path = '/media/lab/Hard Disk/my_datasets/subject7/map1/iter0/2021_Feb_24_14_51_50'

    # Latent NN controller
    img_resize = (64, 64)
    vae_model_path = '/media/lab/Hard Disk/my_outputs/VAE/vanilla_vae/vanilla_vae_model_z_15.pt'
    vae_model_type = 'vanilla_vae'
    latent_model_path = '/media/lab/Hard Disk/my_outputs/subject7/iter0/latent_nn_simple/latent_nn_simple_model_z_15.pt'
    latent_model_type = 'latent_nn_simple'
    controller_agent = LatentCtrl(vae_model_path,
                            vae_model_type,
                            latent_model_path,
                            latent_model_type,
                            img_resize)

    # read data
    num_prvs = 5
    prvs_mode = 'exponential'
    timestamp, yaw_cmd, X, file_list_color = read_data(folder_path, num_prvs, prvs_mode)
    yawRate = X[:,-1]
    cmd_history = X[:,:-1]

    cv2.namedWindow('raw', cv2.WINDOW_NORMAL)

    # Start the loop
    i = 0
    tic = time.perf_counter()
    for color_file in file_list_color:
        image_bgr = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        # pred
        y_pred, z, image_pred, image_raw = controller_agent.predict(image_rgb, yawRate[i], cmd_history[i,:])
        
        # plot
        plot_with_cmd_compare('latent', image_bgr, yaw_cmd[i]/2, y_pred/2)

        image_pred = image_pred.numpy()
        image_raw = image_raw.numpy()
        image_combine = np.concatenate((image_raw, image_pred), axis=2)
        image_combine = ((image_combine + 1.0) / 2.0 * 255.0).astype(np.uint8)
        cv2.imshow('raw', cv2.cvtColor(image_combine.transpose(1,2,0), cv2.COLOR_RGB2BGR))

        elapsed_time = time.perf_counter() - tic
        time.sleep(max(timestamp[i] - elapsed_time, 0))
        i += 1