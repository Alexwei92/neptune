import os
import torch
import glob
import cv2
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import pandas
import time
import multiprocessing as mp
from tqdm import tqdm
import random
import math

# Exponential decaying function
def exponential_decay(num_prvs=5, max_prvs=15, ratio=1.5):
    y = []
    for t in range(0, num_prvs):
        y.append(int(np.ceil(max_prvs * np.exp(-t/ratio))))
    return y

# Check box
def check_box(img):
    height = img.shape[0]
    width = img.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    range_x = (int(w1*width), int(w2*width))
    range_y = (int(h1*height), int(h2*height))

    # top edge
    box_edge_top = img[range_y[0], range_x[0]:range_x[1]]
    box_edge_bottom = img[range_y[1], range_x[0]:range_x[1]]
    box_edge = box_edge_top - box_edge_bottom
    if box_edge.max() == 0:
        return True
    else:
        return False

# Generate a sample folder
def genereate_sample_folder(folder_path, test_dataloader, checkpoint_preload, num_workers=6):
    if not checkpoint_preload and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        _, example_data = next(enumerate(test_dataloader)) 
        if isinstance(example_data, list):
            example_data = example_data[0]
            
        torch.save(example_data, os.path.join(folder_path, 'sample_image_data.pt'))
        vutils.save_image(example_data,
                        os.path.join(folder_path, 'sample_image.png'),
                        normalize=True,
                        range=(-1,1))
    else:
        example_data = torch.load(os.path.join(folder_path, 'sample_image_data.pt'))
    
    return example_data

class ImageDataset_simple(Dataset):
    '''
    Image Dataset Class in numpy (simple version)
    '''
    def __init__(self, folder_path, resize, in_channels=3, preload=True, transform=None):
        self.images_np = np.empty((0, resize[0], resize[1], in_channels), dtype=np.float32)
        self.filename = 'image_data_preload.pt'
        self.transform = transform

        # Configure
        self.configure(folder_path, resize, in_channels, preload)

    def configure(self, folder_path, resize, in_channels, preload):
        jobs = [] # use multi-core
        pool = mp.Pool(min(12, mp.cpu_count())) # use how many cores
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            save_to_filepath = os.path.join(subfolder_path, self.filename)

            if preload and os.path.isfile(save_to_filepath):
                jobs.append(pool.apply_async(torch.load, args=(save_to_filepath,)))
            else:   
                jobs.append(pool.apply_async(self.get_images, args=(subfolder_path, save_to_filepath, resize, in_channels)))

               
        for proc in tqdm(jobs):
            images_np = proc.get()
            self.images_np = np.concatenate((self.images_np, images_np), axis=0)        
    
        
    def get_images(self, subfolder_path, save_to_filepath, resize, in_channels=3):
        file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
        file_list.sort()
        images_np = np.zeros((len(file_list), resize[0], resize[1], in_channels), dtype=np.float32)
        idx = 0
        for file, idx in zip(file_list, range(len(file_list))):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize[1], resize[0]))
            images_np[idx, :] = img[:,:,:in_channels]

        # Save output for future uses
        torch.save(images_np, save_to_filepath)
        return images_np

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.choice(range(len(self.images_np)))
        output = self.images_np[idx, :]
        output2 = self.images_np[idx2, :]
        if self.transform:
            output = self.transform(output)
            output2 = self.transform(output2)

        return output, output2
    
class ImageDataset_advanced(Dataset):
    '''
    Image Dataset Class in numpy (advanced version)
    '''
    def __init__(self, dataset_dir, subject_list, map_list, iter, resize, in_channels=3, preload=True, transform=None):
        self.images_np = np.empty((0, resize[0], resize[1], in_channels), dtype=np.float32)
        self.filename = 'image_data_preload.pt'
        self.transform = transform
        
        # Configure
        self.configure(dataset_dir, subject_list, map_list, iter, resize, in_channels, preload)

    def configure(self, dataset_dir, subject_list, map_list, iter, resize, in_channels, preload):
        jobs = [] # use multi-core
        pool = mp.Pool(min(4, mp.cpu_count())) # use how many cores
        iteration = 'iter' + str(iter)
        for subject in subject_list:
            for map in os.listdir(os.path.join(dataset_dir, subject)):
                if map in map_list:
                    folder_path = os.path.join(dataset_dir, subject, map, iteration)
                    for subfolder in os.listdir(folder_path):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        save_to_filepath = os.path.join(subfolder_path, self.filename)

                        if preload and os.path.isfile(save_to_filepath):
                            jobs.append(pool.apply_async(torch.load, args=(save_to_filepath,)))
                        else:   
                            jobs.append(pool.apply_async(self.get_images, args=(subfolder_path, save_to_filepath, resize, in_channels)))
      
        for proc in tqdm(jobs):
            images_np = proc.get()
            self.images_np = np.concatenate((self.images_np, images_np), axis=0)        

        pool.close()    

    def get_images(self, subfolder_path, save_to_filepath, resize, in_channels=3):
        file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
        file_list.sort()
        images_np = np.zeros((len(file_list), resize[0], resize[1], in_channels), dtype=np.float32)
        idx = 0
        for file, idx in zip(file_list, range(len(file_list))):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize[1], resize[0]))
            images_np[idx, :] = img[:,:,:in_channels]

        # save output for future uses
        torch.save(images_np, save_to_filepath)
        return images_np

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        idx2 = random.choice(range(len(self.images_np)))
        output = self.images_np[idx, :]
        output2 = self.images_np[idx2, :]
        if self.transform:
            output = self.transform(output)
            output2 = self.transform(output2)

        return output, output2


class LatentDataset_simple(Dataset):
    '''
    Latent Variable Dataset Class in numpy (simple version) 
    '''
    def __init__(self, dataset_dir, num_prvs, prvs_mode, resize, in_channels=3, preload=True, transform=None):
        self.images_np = np.empty((0, resize[0], resize[1], in_channels), dtype=np.float32)
        self.output = np.empty((0,), dtype=np.float32)
        self.state_extra = np.empty((0, num_prvs+1), dtype=np.float32)
        self.transform = transform

        # Configure
        self.configure(dataset_dir, resize, num_prvs, prvs_mode, in_channels, preload)

    def configure(self, dataset_dir, resize, num_prvs, prvs_mode, in_channels, preload):
        for subfolder in os.listdir(dataset_dir):
            subfolder_path = os.path.join(dataset_dir, subfolder)

            state_extra, output = self.read_telemetry(subfolder_path, num_prvs, prvs_mode)
            self.state_extra = np.concatenate((self.state_extra, state_extra), axis=0)
            self.output = np.concatenate((self.output, output), axis=0)

            default_filepath = os.path.join(subfolder_path, 'image_data_preload.pt')
            if preload and os.path.isfile(default_filepath):
                images_np = torch.load(default_filepath)
            else:   
                images_np = self.get_images(subfolder_path, default_filepath, resize, in_channels)

            self.images_np = np.concatenate((self.images_np, images_np), axis=0)

        # Data augmentation
        image_np_flip =  np.empty((len(self.images_np), resize[0], resize[1], in_channels), dtype=np.float32)
        for i in range(len(self.images_np)):
            image_np_flip[i,:] = cv2.flip(self.images_np[i,...], 1)
        state_extra_flip = -self.state_extra
        output_flip = -self.output

        self.images_np = np.concatenate((self.images_np, image_np_flip), axis=0)
        self.state_extra = np.concatenate((self.state_extra, state_extra_flip), axis=0)
        self.output = np.concatenate((self.output, output_flip), axis=0)

    def read_telemetry(self, folder_dir, num_prvs, prvs_mode):
        # Read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_dir, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_dir))
            N -= (5 * 10) # remove the last 5 seconds data
            if N < 0:
                return None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy(dtype=np.float32)

         # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(dtype=np.float32), (-1,1))
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0

        if num_prvs > 0:
            # Previous commands with time decaying
            y_prvs = np.zeros((N, num_prvs), dtype=np.float32)
            if prvs_mode == 'exponential':
                prvs_index = exponential_decay(num_prvs)
            elif prvs_mode == 'linear':
                prvs_index = [i for i in reversed(range(1, num_prvs+1))]
            else:
                raise Exception('Unknown prvs_mode {:s}'.format(prvs_index))

            for i in range(N):
                for j in range(num_prvs):
                    y_prvs[i,j] = y[max(i-prvs_index[j], 0)]
            
            # X_extra = [y_prvs, yawRate]
            X_extra = np.concatenate((y_prvs, yawRate_norm), axis=1)
        else:
            X_extra = yawRate_norm
        
        return X_extra, y

    def get_images(self, subfolder_path, save_to_filepath, resize, in_channels=3):
        file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
        file_list.sort()
        images_np = np.zeros((len(file_list), resize[0], resize[1], in_channels), dtype=np.float32)
        idx = 0
        for file, idx in zip(file_list, range(len(file_list))):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize[1], resize[0]))
            images_np[idx, :] = img[:,:,:in_channels]

        # Save output for future uses
        torch.save(images_np, save_to_filepath)        
        return images_np

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images_np[idx, :]
        if self.transform:
            img = self.transform(img)

        return img, self.state_extra[idx, :], self.output[idx]

class LatentDataset_advanced(Dataset):
    '''
    Latent Variable Dataset Class in numpy (advanced version) 
    '''
    def __init__(self, dataset_dir, subject_list, map_list, iter, num_prvs, prvs_mode, resize, in_channels=3, preload=True, transform=None):
        self.subject_list = subject_list
        self.map_list = map_list
        self.iter = iter
        self.images_np = np.empty((0, resize[0], resize[1], in_channels), dtype=np.float32)
        self.output = np.empty((0,), dtype=np.float32)
        self.state_extra = np.empty((0, num_prvs+1), dtype=np.float32)
        self.transform = transform

        # Configure
        self.configure(dataset_dir, resize, num_prvs, prvs_mode, in_channels, preload)

    def configure(self, dataset_dir, resize, num_prvs, prvs_mode, in_channels, preload):
        iteration = 'iter' + str(self.iter)
        for subject in self.subject_list:
            for map in os.listdir(os.path.join(dataset_dir, subject)):
                if map in self.map_list:
                    folder_path = os.path.join(dataset_dir, subject, map, iteration)
                    for subfolder in os.listdir(folder_path):
                        subfolder_path = os.path.join(folder_path, subfolder)

                        state_extra, output = self.read_telemetry(subfolder_path, num_prvs, prvs_mode)
                        self.state_extra = np.concatenate((self.state_extra, state_extra), axis=0)
                        self.output = np.concatenate((self.output, output), axis=0)

                        default_filepath = os.path.join(subfolder_path, 'image_data_preload.pt')
                        if preload and os.path.isfile(default_filepath):
                            images_np = torch.load(default_filepath)
                        else:   
                            images_np = self.get_images(subfolder_path, default_filepath, resize, in_channels)

                        self.images_np = np.concatenate((self.images_np, images_np), axis=0)
        
        # Data augmentation
        image_np_flip =  np.empty((len(self.images_np), resize[0], resize[1], in_channels), dtype=np.float32)
        for i in range(len(self.images_np)):
            image_np_flip[i,:] = cv2.flip(self.images_np[i,:], 1)
        state_extra_flip = -self.state_extra
        output_flip = -self.output

        self.images_np = np.concatenate((self.images_np, image_np_flip), axis=0)
        self.state_extra = np.concatenate((self.state_extra, state_extra_flip), axis=0)
        self.output = np.concatenate((self.output, output_flip), axis=0)

    def read_telemetry(self, folder_dir, num_prvs, prvs_mode):
        # Read telemetry csv       
        telemetry_data = pandas.read_csv(os.path.join(folder_dir, 'airsim.csv'))
        N = len(telemetry_data) - 1 # length of data 

        if telemetry_data.iloc[-1,0] == 'crashed':
            print('Find crashed dataset in {:s}'.format(folder_dir))
            N -= (5 * 10) # remove the last 5 seconds data
            if N < 0:
                return None, None, None

        # Yaw cmd
        y = telemetry_data['yaw_cmd'][:N].to_numpy(dtype=np.float32)

        # Yaw rate
        yawRate = np.reshape(telemetry_data['yaw_rate'][:N].to_numpy(dtype=np.float32), (-1,1))
        yawRate_norm = yawRate * (180.0 / math.pi) / 45.0

        if num_prvs > 0:
            # Previous commands with time decaying
            y_prvs = np.zeros((N, num_prvs), dtype=np.float32)
            if prvs_mode == 'exponential':
                prvs_index = exponential_decay(num_prvs)
            elif prvs_mode == 'linear':
                prvs_index = [i for i in reversed(range(1, num_prvs+1))]
            else:
                raise Exception('Unknown prvs_mode {:s}'.format(prvs_index))

            for i in range(N):
                for j in range(num_prvs):
                    y_prvs[i,j] = y[max(i-prvs_index[j], 0)]
            
            # X_extra = [y_prvs, yawRate]
            X_extra = np.concatenate((y_prvs, yawRate_norm), axis=1)
        else:
            X_extra = yawRate_norm
        
        return X_extra, y

    def get_images(self, subfolder_path, save_to_filepath, resize, in_channels=3):
        file_list = glob.glob(os.path.join(subfolder_path, 'color', '*.png'))
        file_list.sort()
        images_np = np.zeros((len(file_list), resize[0], resize[1], in_channels), dtype=np.float32)
        idx = 0
        for file, idx in zip(file_list, range(len(file_list))):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize[1], resize[0]))
            images_np[idx, :] = img[:,:,:in_channels]

        # Save output for future uses
        torch.save(images_np, save_to_filepath)        
        return images_np

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images_np[idx, :]
        if self.transform:
            img = self.transform(img)

        return img, self.state_extra[idx, :], self.output[idx]