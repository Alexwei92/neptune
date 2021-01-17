import os
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# # Preprocess the raw images
# def preprocess_data(root_dir, res, n_chan):
    
#     files_list = glob.glob(os.path.join(root_dir, 'raw/*.png'))
#     files_list.sort()

#     size_data = len(files_list)
#     images_np = np.zeros((size_data, res[0], res[1], n_chan)).astype(np.uint8)

#     idx = 0
#     for file in files_list:
#         im = Image.open(file)
#         im = im.resize((res[0], res[1]))
#         images_np[idx, :] = np.array(im)

#         idx = idx + 1
#         if idx == size_data:
#             break
    
#     return images_np

# # example
# torch.save(processed_data, os.path.join(root_dir, 'processed/all_data.pt'))

#################################################
# MyDataset Class (numpy)
class MyDataset_np(Dataset):

    def __init__(self, root, res, n_chan):
        self.root = root

        files_list = glob.glob(os.path.join(root, self.raw_folder, '*.png'))
        files_list.sort()
        self.images_np = np.zeros((len(files_list), res[0], res[1], n_chan)).astype(np.float32)
        idx = 0
        for file in files_list:
            im = Image.open(file)
            im = im.resize((res[0], res[1]))
            im = np.array(im)
            im = im / 255.0 * 2.0 - 1.0 # normalize to [-1,1]
            self.images_np[idx, :] = im[:,:,:3]
            idx = idx + 1

            if idx == len(files_list):
                break

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_np = self.images_np[idx, :]
        return image_np.transpose((2,1,0))

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')


# MyDataset Class (PIL)
class MyDataset_PIL(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # if n_chan == 3:
        #     self.mode = 'RGB'
        # else:
        #     self.mode = 'L'

        self.data = torch.load(os.path.join(self.root, self.processed_folder, 'all_data.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]

        # image = Image.fromarray(image, mode=self.mode)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
