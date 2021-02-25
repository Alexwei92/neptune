import setup_path
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *
from models import *
from imitation_learning import *

def imshow_np(axis, img):
    if torch.is_tensor(img):
        img = img.numpy()
    # input image is normalized to [-1,1]
    img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
    axis.imshow(cv2.cvtColor(img.transpose(2,1,0), cv2.COLOR_BGR2RGB))
    # plt.axis('off')

if __name__ == '__main__':

    # dataset_dir = '/media/lab/Hard Disk/my_datasets/peng/river/VAE'
    # output_dir = '/media/lab/Hard Disk/my_outputs/peng/river/VAE'
    dataset_dir = 'E:/my_datasets/peng/river/VAE'
    output_dir = 'E:/my_outputs/peng/river/VAE'

    # Parameters
    z_dim = 25
    img_resize = (64, 64)
    # train_data = torch.load(os.path.join(dataset_dir, 'train_data.pt'))
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load VAE model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    vae_model_filename = 'vae_model_z_25.pt'
    vae_model = MyVAE(z_dim).to(device)
    model = torch.load(os.path.join(output_dir, vae_model_filename))
    vae_model.load_state_dict(model)    

    image_raw = cv2.imread('0000487.png', cv2.IMREAD_UNCHANGED)
    image_resize = cv2.resize(image_raw, (img_resize[1], img_resize[0]))
    image_norm = normalize_image(image_resize)
    image_norm = image_norm.transpose((2,1,0)).astype(np.float32)
    image_tensor = torch.from_numpy(image_norm).unsqueeze(0)
    

    # 
    vars = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(5, len(vars))

    vae_model.eval()
    with torch.no_grad():
        z_raw_gpu = vae_model.get_latent(image_tensor.to(device))
        z_raw_cpu = z_raw_gpu.cpu().numpy()
        for i in range(len(axes)):
            for value, axis in zip(vars, axes[i]):
                z_new_cpu = z_raw_cpu.copy()
                index = i + 0
                z_new_cpu[0, index] += value
                z_new_gpu = torch.from_numpy(z_new_cpu.astype(np.float32)).unsqueeze(0).to(device)
                image_pred = vae_model.decode(z_new_gpu).cpu().squeeze(0)
            
                imshow_np(axis, image_pred)
                if value == 0:
                    axis.set_title('z={:d}'.format(index+1), fontsize=11, color='b')
                else:
                    axis.set_title(value, fontsize=11)
                axis.set_axis_off()
    
    plt.show()
