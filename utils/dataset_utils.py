import os
import torch
import glob
# from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np

# # Preprocess the raw images
# def preprocess_data(root, desired_size, n_chan=3):
    
#     files_list = glob.glob(os.path.join(root, 'raw', '*.png'))
#     files_list.sort()

#     size_data = len(files_list)
#     images_np = np.zeros((size_data, desired_size[0], desired_size[1], n_chan)).astype(np.uint8)

#     idx = 0
#     for file in files_list:
#         im = Image.open(file)
#         im = im.resize((desired_size[0], desired_size[1]))
#         images_np[idx, :] = np.array(im)

#         idx += 1
#         if idx == size_data:
#             break
    
#     return images_np

# # example
# torch.save(processed_data, os.path.join(root_dir, 'processed/all_data.pt'))

def normalize_image(image):
    # Assume input image is uint8 type
    # Normalize it to [-1.0, 1.0]
    return image / 255.0 * 2.0 - 1.0


class ImageDataset(Dataset):
    '''
    Image Dataset Class in numpy
    '''
    def __init__(self, folder_path, resize, n_chan=3):
        self.folder_path = folder_path
        file_list = glob.glob(os.path.join(folder_path, self.color_folder, '*.png'))
        file_list.sort()
        self.images_np = np.zeros((len(file_list), resize[0], resize[1], n_chan)).astype(np.float32)
        idx = 0
        for file in file_list:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (resize[1], resize[0]))
            img = normalize_image(img)
            self.images_np[idx, :] = img[:,:,:n_chan]
            idx += 1

            if idx == len(file_list):
                break

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images_np[idx, :].transpose((2,1,0))

    @property
    def color_folder(self):
        return os.path.join(self.folder_path, 'color')

    @property
    def depth_folder(self):
        return os.path.join(self.folder_path, 'depth')


class LatentDataset(Dataset):
    '''
    Latent Variable Dataset Class in Numpy
    '''
    def __init__(self, folder_path, resize, n_chan=3):
        self.folder_path = folder_path
        file_list = glob.glob(os.path.join(folder_path, self.color_folder, '*.png'))
        file_list.sort()
        self.images_np = np.zeros((len(file_list), resize[0], resize[1], n_chan)).astype(np.float32)
        idx = 0
        for file in file_list:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (resize[1], resize[0]))
            img = normalize_image(img)
            self.images_np[idx, :] = img[:,:,:n_chan]
            idx += 1

            if idx == len(file_list):
                break
        
        # Read telemetry file
        data = np.genfromtxt(os.path.join(self.folder_path, 'airsim.csv'),
                                delimiter=',', skip_header=True).astype(np.float32)

        # TODO: put column # in configuration file
        self.data = data[:,6]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.images_np[idx, :].transpose((2,1,0)), self.data[idx]

    @property
    def color_folder(self):
        return os.path.join(self.folder_path, 'color')