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

# Generate a sample folder
def genereate_sample_folder(folder_path, test_loader, checkpoint_preload, num_workers=6):
    if not checkpoint_preload and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        _, example_data = next(enumerate(test_loader)) 
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

        output = self.images_np[idx, :]
        if self.transform:
            output = self.transform(output)

        return output
    
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
        
        output = self.images_np[idx, :]
        if self.transform:
            output = self.transform(output)

        return output


class LatentDataset(Dataset):
    '''
    Latent Variable Dataset Class in Numpy
    '''
    def __init__(self, folder_path, num_prvs, resize, n_chan=3, preload=True):
        self.folder_path = folder_path
        self.images_np = np.empty((0, resize[0], resize[1], n_chan), dtype=np.float32)
        self.output = np.empty((0,), dtype=np.float32)
        self.state_extra = np.empty((0, num_prvs+1), dtype=np.float32)

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            print(subfolder_path)
            # Read telemetry file
            telemetry_data = pandas.read_csv(os.path.join(subfolder_path, 'airsim.csv'))
            # if telemetry_data.iloc[-1,0] == 'crashed':
                # print(subfolder_path)
                # continue
                
            # yaw cmd
            y = telemetry_data['yaw_cmd'][:-1].to_numpy(dtype=np.float32)
            self.output = np.concatenate((self.output, y), axis=0)
            # Yaw rate
            yawRate = telemetry_data['yaw_rate'][:-1].to_numpy(dtype=np.float32)
            # Previous commands with time decaying
            y_prvs = np.zeros((len(y), num_prvs), dtype=np.float32)
            prvs_index = [5,4,3,2,1]
            for i in range(len(y)):
                for j in range(num_prvs):
                    y_prvs[i,j] = y[max(i-prvs_index[j], 0)]

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
