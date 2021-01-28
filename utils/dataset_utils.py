import os
import torch
import glob
# from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np

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
    def __init__(self, folder_path, cmd_index, resize, n_chan=3, preload=True):
        self.folder_path = folder_path
        self.images_np = np.empty((0, resize[0], resize[1], n_chan), dtype=np.float32)
        self.data = np.empty((0,), dtype=np.float32)

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

            self.images_np = np.concatenate((self.images_np, images_np), axis=0)

            # Read telemetry file
            data = np.genfromtxt(os.path.join(subfolder_path, 'airsim.csv'),
                                    delimiter=',', skip_header=True, dtype=np.float32)
            data = data[:, cmd_index]
            self.data = np.concatenate((self.data, data), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images_np[idx, :].transpose((2,1,0)), self.data[idx]
