"""
Plot the distributions of the latent variables
"""

import setup_path
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from utils import *
from models import *
from imitation_learning import *

if __name__ == '__main__':

    dataset_dir = '/media/lab/Hard Disk/' + 'my_datasets/peng/river/VAE'
    output_dir = '/media/lab/Hard Disk/' + 'my_outputs/peng/river/VAE'

    # Parameters
    z_dim = 10
    batch_size = 625
    img_resize = (64, 64)

    # Load the data
    transform_composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
        ])
    all_data = ImageDataset_simple(dataset_dir, resize=img_resize, preload=True, transform=transform_composed)
    train_data, _ = train_test_split(all_data, test_size=0.1, random_state=11)        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load VAE model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model_filename = 'vae_model_z_10.pt'
    vae_model = MyVAE(img_resize[0], z_dim).to(device)
    model = torch.load(os.path.join(output_dir, model_filename))
    vae_model.load_state_dict(model)    

    z_result = np.empty((len(train_loader), z_dim), dtype=np.float32)
    vae_model.eval()
    with torch.no_grad():
        for batch_idx, batch_x in enumerate(tqdm(train_loader)):            
            z = vae_model.get_latent(batch_x.to(device))
            z_result = np.concatenate((z_result, z.cpu().numpy()))
    
    # Plot the probability distribution of latent variables
    fig, axes = plt.subplots(int(z_dim/5), 5, sharex=True, sharey=True)
    i = 0
    for axis in axes.flat:
        axis.hist(z_result[:,i], bins=50, range=(-5,5), density=True, color='b', alpha=0.8)
        axis.set_title('z[{:d}]'.format(i+1))
        i += 1
    
    plt.show()