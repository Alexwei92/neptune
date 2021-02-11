import os
import torch
import glob
# from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
import pandas

def normalize_image(image):
    # Assume input image is uint8 type
    # Normalize it to [-1.0, 1.0]
    return image / 255.0 * 2.0 - 1.0

class ImageDataset(Dataset):
    '''
    Image Dataset Class in numpy
    '''
    def __init__(self, folder_path, resize, n_chan=3, preload=True):
        self.folder_path = folder_path
        self.images_np = np.empty((0, resize[0], resize[1], n_chan), dtype=np.float32)

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            print(subfolder_path)
            default_filepath = os.path.join(subfolder_path, 'image_data_preload.pt')
            if preload and os.path.isfile(default_filepath):
                images_np = torch.load(default_filepath)
            else:   
                file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
                file_list.sort()
                images_np = np.zeros((len(file_list), resize[0], resize[1], n_chan), dtype=np.float32)
                idx = 0
                for file in file_list:
                    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (resize[1], resize[0]))
                    img = normalize_image(img)
                    images_np[idx, :] = img[:,:,:n_chan]
                    idx += 1
                    if idx == len(file_list):
                        break
                
                torch.save(images_np, default_filepath)
                
            self.images_np = np.concatenate((self.images_np, images_np), axis=0)

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images_np[idx, :].transpose((2,1,0))

class LatentDataset(Dataset):
    '''
    Latent Variable Dataset Class in Numpy
    '''
    def __init__(self, folder_path, num_prvs, resize, n_chan=3, preload=True):
        self.folder_path = folder_path
        self.images_np = np.empty((0, resize[0], resize[1], n_chan), dtype=np.float32)
        self.output = np.empty((0,), dtype=np.float32)
        self.state_extra = np.empty((0, cmd_numprvs+1), dtype=np.float32)

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            print(subfolder_path)
            # Read telemetry file
            telemetry_data = pandas.read_csv(os.path.join(subfolder_path, 'airsim.csv'))
            if telemetry_data.iloc[-1,0] == 'crashed':
                # print(subfolder_path)
                continue
                
            # yaw cmd
            y = telemetry_data['yaw_cmd'][:-1].to_numpy(dtype=np.float32)
            self.output = np.concatenate((self.output, y), axis=0)
            # Yaw rate
            yawRate = telemetry_data['yaw_rate'][:-1].to_numpy(dtype=np.float32)
            # Previous commands with time decaying
            y_prvs = np.zeros((len(y), num_prvs), dtype=np.float32)
            for i in range(len(y)):
                for j in range(num_prvs):
                    if i > j:
                        y_prvs[i,j] = y[i-(j+1)] * 0.8**(j+1)

            self.state_extra = np.concatenate((self.state_extra, np.column_stack((yawRate, y_prvs))), axis=0)

            default_filepath = os.path.join(subfolder_path, 'image_data_preload.pt')
            if preload and os.path.isfile(default_filepath):
                images_np = torch.load(default_filepath)
            else:   
                file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
                file_list.sort()
                images_np = np.zeros((len(file_list), resize[0], resize[1], n_chan), dtype=np.float32)
                idx = 0
                for file in file_list:
                    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (resize[1], resize[0]))
                    img = normalize_image(img)
                    images_np[idx, :] = img[:,:,:n_chan]
                    idx += 1
                    if idx == len(file_list):
                        break

            self.images_np = np.concatenate((self.images_np, images_np), axis=0)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images_np[idx, :].transpose((2,1,0)), self.state_extra[idx, :], self.output[idx]
