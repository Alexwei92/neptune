import setup_path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from configs import 
from utils import *
from models import *
from imitation_learning import *

def imshow_np(axis, img):
    if torch.is_tensor(img):
        img = img.numpy()
    # input image is normalized to [-1.0,1.0]
    img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
    # input image is normalized to [0.0,1.0]
    # img = (img * 255.0).astype(np.uint8)
    axis.imshow(img.transpose(2,1,0))
    # plt.axis('off')


if __name__ == '__main__':

    dataset_dir = '/media/lab/Hard Disk/my_datasets/test'
    output_dir = '/media/lab/Hard Disk/my_outputs/test2'

    # Parameters
    model_type = 'vanilla_vae'


    z_dim = 10
    img_resize = (64, 64)

    # Load VAE model
    device = torch.device("cuda:0")
    model_filename = 'vae_model_z_10.pt'
    vae_model = MyVAE(img_resize[0], z_dim).to(device)
    model = torch.load(os.path.join(output_dir, model_filename))
    vae_model.load_state_dict(model)    

    image_raw = cv2.imread('0000900.png', cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image_rgb, (img_resize[1], img_resize[0]))

    transform_composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
        ])


    # image_tensor = torch.from_numpy(image_norm).unsqueeze(0)
    image_tensor = transform_composed(image_resize).unsqueeze(0)
    
    # Perturbation
    vars = [-2.0, -1.0, 0.0, 1.0, 2.0] 
    fig, axes = plt.subplots(5, len(vars))

    vae_model.eval()
    with torch.no_grad():
        z_raw_gpu = vae_model.get_latent(image_tensor.to(device))
        z_raw_cpu = z_raw_gpu.cpu().numpy()
        for i in range(len(axes)):
            for value, axis in zip(vars, axes[i]):
                z_new_cpu = z_raw_cpu.copy()
                index = i + 5
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
