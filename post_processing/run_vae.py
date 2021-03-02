import setup_path
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import *
from models import *
from imitation_learning import *

if __name__ == '__main__':

    dataset_dir = '/media/lab/Hard Disk/my_datasets/peng/river/VAE'
    output_dir = '/media/lab/Hard Disk/my_outputs/peng/river/VAE'
    # dataset_dir = 'E:/my_datasets/peng/river/VAE'
    # output_dir = 'E:/my_outputs/peng/river/VAE'

    # Parameters
    z_dim = 15
    batch_size = 625
    img_resize = (64, 64)
    train_data = torch.load(os.path.join(dataset_dir, 'train_data.pt'))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load VAE model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    vae_model_filename = 'vae_model_z_10_new.pt'
    vae_model = MyVAE(z_dim).to(device)
    model = torch.load(os.path.join(output_dir, vae_model_filename))
    vae_model.load_state_dict(model)    

    z_result = np.empty((len(train_loader), z_dim), dtype=np.float32)
    vae_model.eval()
    with torch.no_grad():
        for batch_idx, batch_x in enumerate(tqdm(train_loader)):            
            z = vae_model.get_latent(batch_x.to(device))
            z_result = np.concatenate((z_result, z.cpu().numpy()))
    
    # Plot
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
    i = 0
    for axis in axes.flat:
        axis.hist(z_result[:,i], bins=30, range=(-3,3), density=True, color='b', alpha=0.8)
        axis.set_title('z[{:d}]'.format(i+1))
        i += 1
    
    plt.show()