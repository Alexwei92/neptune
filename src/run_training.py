import setup_path
import glob
import os
import shutil
import yaml
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision import transforms
import torch

from utils import *
from models import *
from imitation_learning import *

# torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    # Read YAML configurations
    try:
        file = open('training_config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    # Training settings
    folder_path = config['train_params']['folder_path']    
    if len(folder_path) == 0: # if leave it ''
        dataset_dir = os.path.join(setup_path.parent_dir, config['train_params']['dataset_dir']) 
        output_dir = os.path.join(setup_path.parent_dir, config['train_params']['output_dir']) 
    else:
        dataset_dir = os.path.join(folder_path, config['train_params']['dataset_dir']) 
        output_dir = os.path.join(folder_path, config['train_params']['output_dir']) 

    if not os.path.isdir(dataset_dir):
        raise Exception("No such folder {:s}".format(dataset_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    train_reg = config['train_params']['train_reg']    
    train_vae = config['train_params']['train_vae']  
    train_latent = config['train_params']['train_latent']  
    result_only = config['train_params']['result_only']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    if result_only:
        # Only display results, no training
        train_reg, train_vae, train_latent = False, False, False

        # print('\n***** Regression Results *****')
        # reg_model_filename = config['train_params']['reg_model_filename']
        # reg_weight_filename = config['train_params']['reg_weight_filename']
        # reg_result = pickle.load(open(os.path.join(output_dir, reg_model_filename), 'rb'))
        
        # print('Regression type = {:s}'.format(str(reg_result['Model'])))
        # print('R_square = {:.6f}'.format(reg_result['R_square']))
        # print('RMSE = {:.6f}'.format(reg_result['RMSE']))
        # print('Number of weights = {:} '.format(len(reg_result['Model'].coef_)+1))
        # print('****************************\n')

        print('\n***** VAE Results *****')
        model_type = config['train_params']['model_type']
        img_resize = eval(config['train_params']['img_resize'])
        dataloader_type = config['train_params']['dataloader_type']
        transform_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
        ])
        if dataloader_type == 'simple':
            all_data = ImageDataset_simple(dataset_dir, resize=img_resize, preload=True, transform=transform_composed)
        elif dataloader_type == 'advanced':
            all_data = ImageDataset_advanced(dataset_dir, subject_list, map_list, iter=0, resize=img_resize, preload=True, transform=transform_composed)
        else:
            raise Exception("Unknown dataloader_type {:s}".format(dataloader_type))
        
        _, test_data = train_test_split(all_data, test_size=0.1, random_state=11)

        print('Model Type: {:s}'.format(model_type))
        if model_type == 'vae':
            z_dim = config['train_params']['vae_z_dim']
            batch_size = 64
            checkpoint_filename = config['train_params']['vae_checkpoint_filename']
            vae_agent = VAETrain(MyVAE(img_resize[0], z_dim))
            vae_agent.load_checkpoint(os.path.join(output_dir, checkpoint_filename))
            vae_agent.plot_train_result()
            print('Epoch = {:d}'.format(vae_agent.epoch[-1]))

            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=6)
            vae_agent.test(test_loader)
            with torch.no_grad(): 
                _, example_data = next(enumerate(test_loader)) 
                generated_data, _, _ = vae_agent.VAE_model(example_data.to(device))
                plot_generate_figure(example_data, generated_data.cpu()) 
        
        elif model_type == 'vaegan':
            z_dim = config['train_params']['vaegan_z_dim']
            batch_size = 64
            checkpoint_filename = config['train_params']['vaegan_checkpoint_filename']
            vaegan_agent = VAEGANTrain(z_dim, Encoder(z_dim), Generator(z_dim), Discriminator())
            vaegan_agent.load_checkpoint(os.path.join(output_dir, checkpoint_filename))

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            ax1.plot(vaegan_agent.epoch, vaegan_agent.loss_E, color='blue')
            ax1.legend(['Encoder Loss'], loc='upper right')
            ax2.plot(vaegan_agent.epoch, vaegan_agent.loss_G, color='blue')
            ax2.legend(['Generator Loss'], loc='upper right')
            # ax2.set_ylabel('Loss')
            ax3.plot(vaegan_agent.epoch, vaegan_agent.loss_D, color='blue')
            ax3.legend(['Discriminator Loss'], loc='upper right')
            ax4.plot(vaegan_agent.epoch, vaegan_agent.loss_kld, color='blue')
            ax4.legend(['KLD Loss'], loc='upper right')        
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Epoch')
            # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        
        else:
            raise Exception("Unknown model_type {:s}".format(model_type))

        # print('\n***** NN Controller Results *****')
        # latent_checkpoint_filename = config['train_params']['latent_checkpoint_filename']
        # latent_num_prvs = config['train_params']['latent_num_prvs']
        # latent_agent = LatentTrain(MyLatent(z_dim+latent_num_prvs+1), MyVAE(z_dim))
        # latent_agent.load_checkpoint(os.path.join(output_dir, latent_checkpoint_filename))
        # latent_agent.plot_train_result()
        # print('****************************\n')

        plt.show()

    # 1) Linear Regression
    if train_reg:
        print('======= Linear Regression Controller =========')
        image_size = eval(config['ctrl_params']['image_size'])
        preload_sample = config['train_params']['preload_sample']
        reg_num_prvs = config['train_params']['reg_num_prvs']
        reg_type = config['train_params']['reg_type']
        reg_weight_filename = config['train_params']['reg_weight_filename']
        reg_model_filename = config['train_params']['reg_model_filename']
        use_multicore = config['train_params']['use_multicore']

        reg_kwargs = {
            'dataset_dir': dataset_dir,
            'output_dir': output_dir,
            'image_size': image_size,
            'num_prvs': reg_num_prvs,
            'reg_type': reg_type,
            'weight_filename': reg_weight_filename,
            'model_filename': reg_model_filename,
            'preload': preload_sample,
            'printout': False,
        }

        if use_multicore: 
            # Multi-processing training
            proc = mp.Process(target=RegTrain_multi, kwargs=reg_kwargs)
            proc.start()
            proc.join()
        else:
            # Single-processing training
            RegTrain_single(**reg_kwargs)

    # 2) VAE
    if train_vae:
        print('============== VAE model ================')
        # Subject list
        subject_list = [
            'subject1',
            'subject2',
            'subject3',
            'subject4',
            'subject5',
            'subject6',
            'subject7',
            'subject8',
            'subject9',
            'subject10',
        ]
        # Map list
        map_list = [
            'map1',
            'map2',
            'map3',
            'map4',
            'map5',
            'map7',
            'map8',
            # 'map9',
            # 'o1',
            'o2',
        ]

        # DataLoader
        print('Loading datasets...')
        dataloader_type = config['train_params']['dataloader_type']
        img_resize = eval(config['train_params']['img_resize'])
        transform_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
        ])

        if dataloader_type == 'simple':
            all_data = ImageDataset_simple(dataset_dir, resize=img_resize, preload=True, transform=transform_composed)
        elif dataloader_type == 'advanced':
            all_data = ImageDataset_advanced(dataset_dir, subject_list, map_list, iter=0, resize=img_resize, preload=True, transform=transform_composed)
        else:
            raise Exception("Unknown dataloader_type {:s}".format(dataloader_type))
        
        train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=11)        
        print('Load VAE datasets successfully!')

        # Model
        model_type = config['train_params']['model_type']
        if model_type == 'vae':
            # VAE
            z_dim = config['train_params']['vae_z_dim']
            batch_size = config['train_params']['vae_batch_size']
            n_epochs = config['train_params']['vae_n_epochs']
            checkpoint_filename = config['train_params']['vae_checkpoint_filename']
            checkpoint_preload = config['train_params']['vae_checkpoint_preload']
            model_filename = config['train_params']['vae_model_filename']
            learning_rate = config['train_params']['vae_learning_rate']
            generate_samples = config['train_params']['vae_generate_samples']

            # Create sample folder
            if generate_samples:
                N_sample = 64
                test_loader = DataLoader(test_data, batch_size=N_sample, shuffle=True, num_workers=6)
                sample_folder = os.path.join(output_dir, 'vae_sample_folder_z_' + str(z_dim))
                if not checkpoint_preload and os.path.isdir(sample_folder):
                    shutil.rmtree(sample_folder)
                if not os.path.isdir(sample_folder):
                   os.makedirs(sample_folder)
                   _, example_data = next(enumerate(test_loader)) 
                   torch.save(example_data, os.path.join(sample_folder, 'sample_image_data.pt'))
                   vutils.save_image(example_data,
                                    os.path.join(sample_folder, 'sample_image.png'),
                                    normalize=True,
                                    range=(-1,1))
                else:
                    example_data = torch.load(os.path.join(sample_folder, 'sample_image_data.pt'))

            # Create the agent
            vae_agent = VAETrain(MyVAE(img_resize[0], z_dim), learning_rate)
            if checkpoint_preload:
                vae_agent.load_checkpoint(os.path.join(output_dir, checkpoint_filename))

            # Training loop
            print('\n*** Start training ***')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
            for epoch in tqdm(range(1, n_epochs + 1)):
                vae_agent.train(epoch, train_loader)

                if epoch % 5 == 0: # save the data every 5 epochs
                    vae_agent.save_checkpoint(epoch, os.path.join(output_dir, checkpoint_filename))
                    vae_agent.save_model(os.path.join(output_dir, model_filename))
                    if generate_samples:
                        vae_agent.VAE_model.eval()
                        with torch.no_grad(): 
                            generated_sample, _, _ = vae_agent.VAE_model(example_data.to(device)) 
                            vutils.save_image(generated_sample,
                                    os.path.join(sample_folder, 'generated_image_epoch_{:d}.png'.format(vae_agent.get_current_epoch())), 
                                    normalize=True,
                                        range=(-1,1))
            
            print('Trained VAE model successfully.')

        elif model_type == 'vaegan':
            # VAE + GAN
            z_dim = config['train_params']['vaegan_z_dim']
            n_epochs = config['train_params']['vaegan_n_epochs']
            batch_size = config['train_params']['vaegan_batch_size']
            checkpoint_filename = config['train_params']['vaegan_checkpoint_filename']
            checkpoint_preload = config['train_params']['vaegan_checkpoint_preload']
            model_filename = config['train_params']['vaegan_model_filename']
            learning_rate = config['train_params']['vaegan_learning_rate']
            generate_samples = config['train_params']['vaegan_generate_samples']

            # Create sample folder
            if generate_samples:
                N_sample = 64
                test_loader = DataLoader(test_data, batch_size=N_sample, shuffle=True, num_workers=6)
                sample_folder = os.path.join(output_dir, 'vaegan_sample_folder_z_' + str(z_dim))
                if not checkpoint_preload and os.path.isdir(sample_folder):
                    shutil.rmtree(sample_folder)
                if not os.path.isdir(sample_folder):
                   os.makedirs(sample_folder)
                   _, example_data = next(enumerate(test_loader)) 
                   torch.save(example_data, os.path.join(sample_folder, 'sample_image_data.pt'))
                   vutils.save_image(example_data,
                                    os.path.join(sample_folder, 'sample_image.png'),
                                    normalize=True,
                                    range=(-1,1))
                else:
                    example_data = torch.load(os.path.join(sample_folder, 'example_data.pt'))

            # Create the agent
            # vaegan_agent = VAEGANTrain(z_dim, Encoder(z_dim), Generator(z_dim), Discriminator())
            vaegan_agent = VAEGANTrain(MyVAEGAN(img_resize[0], z_dim), z_dim, learning_rate)
            if checkpoint_preload:
                vaegan_agent.load_checkpoint(os.path.join(output_dir, checkpoint_filename))

            # Training loop
            print('\n*** Start training ***')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
            for epoch in tqdm(range(1, n_epochs + 1)):
                vaegan_agent.train(epoch, train_loader)

                if epoch % 1 == 0: # save the data every 5 epochs
                    vaegan_agent.save_checkpoint(epoch, os.path.join(output_dir, checkpoint_filename))
                    # vaegan_agent.save_model(os.path.join(output_dir, model_filename))
                    if generate_samples:
                        vaegan_agent.VAEGAN_model.eval()
                        with torch.no_grad(): 
                            generated_data = vaegan_agent.VAEGAN_model(example_data.to(device))
                            vutils.save_image(generated_data,
                                    os.path.join(sample_folder, 'generated_image_epoch_{:d}.png'.format(vaegan_agent.get_current_epoch())), 
                                    normalize=True,
                                    range=(-1,1))

            print('Trained VAEGAN model successfully.')        

        elif model_type == 'gan':
            # VAE
            z_dim = config['train_params']['gan_z_dim']
            batch_size = config['train_params']['gan_batch_size']
            n_epochs = config['train_params']['gan_n_epochs']
            checkpoint_filename = config['train_params']['gan_checkpoint_filename']
            checkpoint_preload = config['train_params']['gan_checkpoint_preload']
            model_filename = config['train_params']['gan_model_filename']
            learning_rate = config['train_params']['gan_learning_rate']
            generate_samples = config['train_params']['gan_generate_samples']

            # Create sample folder
            if generate_samples:
                N_sample = 64
                test_loader = DataLoader(test_data, batch_size=N_sample, shuffle=True, num_workers=6)
                sample_folder = os.path.join(output_dir, 'gan_sample_folder_z_' + str(z_dim))
                if not checkpoint_preload and os.path.isdir(sample_folder):
                    shutil.rmtree(sample_folder)
                if not os.path.isdir(sample_folder):
                   os.makedirs(sample_folder)
                   _, example_data = next(enumerate(test_loader)) 
                   torch.save(example_data, os.path.join(sample_folder, 'sample_image_data.pt'))
                   vutils.save_image(example_data,
                                    os.path.join(sample_folder, 'sample_image.png'),
                                    normalize=True,
                                    range=(-1,1))
                else:
                    example_data = torch.load(os.path.join(sample_folder, 'sample_image_data.pt'))

            # Create the agent
            # vae_agent = VAETrain(MyVAE(img_resize[0], z_dim), learning_rate)
            netG = Generator(z_dim, img_resize[0])
            netD = Discriminator(img_resize[0])
            gan_agent = GANTrain(netG, netD, z_dim, learning_rate)
            if checkpoint_preload:
                gan_agent.load_checkpoint(os.path.join(output_dir, checkpoint_filename))

            # Training loop
            print('\n*** Start training ***')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
            for epoch in tqdm(range(1, n_epochs + 1)):
                gan_agent.train(epoch, train_loader)

                if epoch % 1 == 0: # save the data every 5 epochs
                    gan_agent.save_checkpoint(epoch, os.path.join(output_dir, checkpoint_filename))
                    # gan_agent.save_model(os.path.join(output_dir, model_filename))
                    if generate_samples:
                        gan_agent.netG.eval()
                        gan_agent.netD.eval()
                        with torch.no_grad(): 
                            noise = torch.randn(64, z_dim, 1, 1, device=device)
                            batch_fake = gan_agent.netG(noise)
                            vutils.save_image(batch_fake,
                                    os.path.join(sample_folder, 'generated_image_epoch_{:d}.png'.format(gan_agent.get_current_epoch())), 
                                    normalize=True,
                                    range=(-1,1))
            
            print('Trained GAN model successfully.')        
        
        else:
            raise Exception("Unknown model_type {:s}".format(model_type))

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
        latent_num_prvs = config['train_params']['latent_num_prvs']
        latent_learning_rate = config['train_params']['latent_learning_rate']
        vae_model_filename = config['train_params']['vae_model_filename']
        
        # DataLoader
        print('Loading latent datasets...')
        data = LatentDataset(dataset_dir, num_prvs=latent_num_prvs, resize=img_resize, preload=False)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)
        print('Load latent datasets successfully.')

        # Create agent
        latent_agent = LatentTrain(MyLatent(z_dim+latent_num_prvs+1), MyVAE(z_dim), latent_learning_rate)
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