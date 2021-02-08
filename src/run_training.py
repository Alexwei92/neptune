import setup_path
import glob
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import *
from models import *
from imitation_learning import *

# torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    # Read YAML configurations
    try:
        file = open('config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    # Training settings
    dataset_dir = os.path.join(setup_path.parent_dir, config['train_params']['dataset_dir']) 
    output_dir = os.path.join(setup_path.parent_dir, config['train_params']['output_dir']) 
    
    if not os.path.isdir(dataset_dir):
        raise Exception("No such folder {:s}".format(dataset_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    train_reg = config['train_params']['train_reg']    
    train_vae = config['train_params']['train_vae']  
    train_latent = config['train_params']['train_latent']  
    result_only = config['train_params']['result_only']

    if result_only:
        # Only display results, no training
        train_reg, train_vae, train_latent = False, False, False

        batch_size = config['train_params']['vae_batch_size']
        img_resize = eval(config['train_params']['img_resize'])
        all_data = ImageDataset(dataset_dir, resize=img_resize, preload=True)
        _, test_data = train_test_split(all_data, test_size=0.1, random_state=11) # split into train and test datasets
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=6)

        z_dim = config['train_params']['z_dim']
        vae_checkpoint_filename = config['train_params']['vae_checkpoint_filename']
        vae_agent = VAETrain(MyVAE(z_dim))
        vae_agent.load_checkpoint(os.path.join(output_dir, vae_checkpoint_filename))
        vae_agent.plot_train_result()

        examples = enumerate(test_loader) 
        batch_idx, example_data = next(examples) 
        with torch.no_grad(): 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            generated_data, _, _ = vae_agent.VAE_model(example_data.to(device)) 
            plot_generate_figure(generated_data.cpu(), example_data, N=6) 

        latent_checkpoint_filename = config['train_params']['latent_checkpoint_filename']
        latent_agent = LatentTrain(MyLatent(z_dim), MyVAE(z_dim))
        latent_agent.load_checkpoint(os.path.join(output_dir, latent_checkpoint_filename))
        latent_agent.plot_train_result()

        plt.show()

    # cmd location in telemetry data
    # cmd_index = config['train_params']['cmd_index']

    # 1) Linear Regression
    if train_reg:
        print('======= Linear Regression Controller =========')
        image_size = eval(config['ctrl_params']['image_size'])
        preload_sample = config['train_params']['preload_sample']
        
        # Create the agent
        # tic = time.perf_counter()
        # tmp1 = mp.Process(target=RegTrain_single, args=(dataset_dir, image_size, preload_sample, False))
        # reg_agent1 = RegTrain_single(dataset_dir, image_size, preload_sample, False)
        # tmp1.start()
        # tmp1.join()
        # print('Single core time: ', time.perf_counter() - tic)
        # reg_agent1.train()

        tic = time.perf_counter()
        tmp2 = mp.Process(target=RegTrain_multi, args=(dataset_dir, image_size, preload_sample, False))
        #reg_agent2 = RegTrain_multi(dataset_dir, image_size, preload_sample, False)
        tmp2.start()
        tmp2.join()
        print('Multi core time: ', time.perf_counter() - tic)

        # Train the model
        #reg_agent2.train()
        print('Trained linear regression model successfully.')

        # Save weight to file
        # reg_agent2.save_weight(os.path.join(output_dir, 'reg_weight.csv'))

    # 2) VAE
    if train_vae:
        print('============== VAE model ================')
        batch_size = config['train_params']['vae_batch_size']
        n_epochs = config['train_params']['vae_n_epochs']
        z_dim = config['train_params']['z_dim']
        img_resize = eval(config['train_params']['img_resize'])
        vae_checkpoint_filename = config['train_params']['vae_checkpoint_filename']
        vae_checkpoint_preload = config['train_params']['vae_checkpoint_preload']
        vae_model_filename = config['train_params']['vae_model_filename']

        # DataLoader
        print('Loading VAE datasets...')
        all_data = ImageDataset(dataset_dir, resize=img_resize, preload=True)
        train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=11) # split into train and test datasets

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=6)
        print('Load VAE datasets successfully.')

        # Create the agent
        vae_agent = VAETrain(MyVAE(z_dim))
        if vae_checkpoint_preload:
            vae_agent.load_checkpoint(os.path.join(output_dir, vae_checkpoint_filename))

        # Training loop
        for epoch in range(1, n_epochs + 1):
            vae_agent.train(epoch, train_loader)
            vae_agent.test(test_loader)

            if epoch % 10 == 0 and epoch > 0:
                vae_agent.save_checkpoint(epoch, os.path.join(output_dir, vae_checkpoint_filename))
                vae_agent.save_model(os.path.join(output_dir, vae_model_filename))
        print('Trained VAE model successfully.')
         
    # 3) Controller Network
    if train_latent:
        print('============= Latent Controller ==============')
        batch_size = config['train_params']['latent_batch_size']
        n_epochs = config['train_params']['latent_n_epochs']
        z_dim = config['train_params']['z_dim']
        img_resize = eval(config['train_params']['img_resize'])
        latent_checkpoint_filename = config['train_params']['latent_checkpoint_filename']
        latent_checkpoint_preload = config['train_params']['latent_checkpoint_preload']
        latent_model_filename = config['train_params']['latent_model_filename']
        vae_model_filename = config['train_params']['vae_model_filename']

        # DataLoader
        print('Loading latent datasets...')
        data = LatentDataset(dataset_dir, cmd_index=cmd_index, resize=img_resize, preload=True)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)
        print('Load latent datasets successfully.')

        # Create agent
        latent_agent = LatentTrain(MyLatent(z_dim), MyVAE(z_dim))
        latent_agent.load_VAEmodel(os.path.join(output_dir, vae_model_filename))
        if latent_checkpoint_preload:
            latent_agent.load_checkpoint(os.path.join(output_dir, latent_checkpoint_filename))

        # Training loop
        for epoch in range(1, n_epochs + 1):
            latent_agent.train(epoch, data_loader)
            # latent_agent.test(test_loader)

            if epoch % 10 == 0 and epoch > 0:
                latent_agent.save_checkpoint(epoch, os.path.join(output_dir, latent_checkpoint_filename))
                latent_agent.save_model(os.path.join(output_dir, latent_model_filename))
        print('Trained Latent model successfully.')   