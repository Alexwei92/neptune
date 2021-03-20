import setup_path
import yaml
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

from utils import *
from models import *
from imitation_learning import *

# torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

def imshow_np(axis, img):
    if torch.is_tensor(img):
        img = img.numpy()
    # input image is normalized to [-1.0,1.0]
    img = (img * 255.0).astype(np.uint8)
    # input image is normalized to [0.0,1.0]
    # img = (img * 255.0).astype(np.uint8)
    axis.imshow(img.transpose(2,1,0))
    # plt.axis('off')

if __name__ == '__main__':

    # Read YAML configurations
    try:
        file = open('training_config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    # Path settings
    folder_path = config['global_params']['folder_path']    
    if len(folder_path) == 0:  # if leave it '', refer to the current folder
        dataset_dir = os.path.join(setup_path.parent_dir, config['global_params']['dataset_dir']) 
        output_dir = os.path.join(setup_path.parent_dir, config['global_params']['output_dir']) 
    else:  # otherwise, refer to the specified folder
        dataset_dir = os.path.join(folder_path, config['global_params']['dataset_dir']) 
        output_dir = os.path.join(folder_path, config['global_params']['output_dir']) 

    if not os.path.isdir(dataset_dir):
        raise Exception("No such folder {:s}".format(dataset_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # CUDA device
    device = torch.device(config['train_params']['device']) 

    # Dataloader
    img_resize = eval(config['train_params']['img_resize'])
    dataloader_type = config['dataset_params']['dataloader_type']
    if dataloader_type == 'advanced':
        subject_list = config['dataset_params']['subject_list']
        map_list = config['dataset_params']['map_list']

    transform_composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)), # from [0,255] to [-1,1]
    ])    


    # If only displays the result
    result_only = config['global_params']['result_only']
    if result_only:
        # Only display results, no training

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
        try:
            file = open(os.path.join(setup_path.parent_dir, 'configs', model_type + '.yaml'), 'r')
            model_config = yaml.safe_load(file)
            file.close()
        except Exception as error:
            print_msg(str(error), type=3)
            exit()

        if dataloader_type == 'simple':
            all_data = ImageDataset_simple(dataset_dir, resize=img_resize, transform=transform_composed)
        elif dataloader_type == 'advanced':
            all_data = ImageDataset_advanced(dataset_dir, subject_list, map_list, iter=0, resize=img_resize, transform=transform_composed)
        else:
            raise Exception("Unknown dataloader_type {:s}".format(dataloader_type))
        
        _, test_data = train_test_split(all_data,
                                    test_size=config['dataset_params']['test_size'],
                                    random_state=config['dataset_params']['manual_seed'])

        print('Model Type: {:s}'.format(model_type))
        if model_type in vae_model:
            model = vae_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = VAETrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

            # plot_train_losses(test_agent.get_train_history())
            # plot_KLD_losses(test_agent.iteration, test_agent.kld_losses_z, plot_sum=False)
            print('Epoch = {:d}'.format(test_agent.get_current_epoch()))


            # Perturbation
            vars = np.linspace(-3.0, 3.0, num=10)
            img_all = torch.empty((1,3,64,64),requires_grad=False).to(device)

            test_agent.model.eval()
            with torch.no_grad(): 
                img = test_data[50]
                z_raw_gpu = test_agent.model.get_latent(img.unsqueeze(0).to(device))
                z_raw_cpu = z_raw_gpu.cpu().numpy()
                for i in range(test_agent.z_dim):
                    z_new_cpu = z_raw_cpu.copy()
                    for value in vars:
                        z_new_cpu[0, i] = value
                        z_new_gpu = torch.from_numpy(z_new_cpu.astype(np.float32)).unsqueeze(0).to(device)
                        img_new_gpu = test_agent.model.decode(z_new_gpu)
                        img_all = torch.cat((img_all, img_new_gpu), axis=0)
                        image_pred = img_new_gpu.cpu().squeeze(0)

            vutils.save_image(img_all[1:,...].cpu(),
                        os.path.join(output_dir, model_type + '_traveler.png'),
                        nrow=len(vars),
                        normalize=True,
                        range=(-1,1))
            img_grid = vutils.make_grid(img_all[1:,...].cpu(), nrow=len(vars), normalize=True, range=(-1,1))
            plt.imshow(img_grid.permute(1,2,0))
            plt.axis('off')
            plt.show()

            # print(test_data[50])
            # test_agent.test(test_data)
            # with torch.no_grad(): 
            #     _, example_data = next(enumerate(test_loader)) 
            #     generated_data, _, _ = test_agent.VAE_model(example_data.to(device))
            #     plot_generate_figure(example_data, generated_data.cpu())
        
        elif model_type in vaegan_model:
            model = vaegan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = VAEGANTrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])


            plot_KLD_losses(test_agent.iteration, test_agent.kld_losses_dim_wise, plot_sum=False)
            print('Epoch = {:d}'.format(test_agent.get_current_epoch()))

            # # Perturbation
            # vars = np.linspace(-3.0, 3.0, num=10)
            # img_all = torch.empty((1,3,64,64),requires_grad=False).to(device)

            # test_agent.model.eval()
            # with torch.no_grad(): 
            #     img = test_data[50]
            #     z_raw_gpu = test_agent.model.get_latent(img.unsqueeze(0).to(device))
            #     z_raw_cpu = z_raw_gpu.cpu().numpy()
            #     for i in range(test_agent.z_dim):
            #         z_new_cpu = z_raw_cpu.copy()
            #         for value in vars:
            #             z_new_cpu[0, i] = value
            #             z_new_gpu = torch.from_numpy(z_new_cpu.astype(np.float32)).unsqueeze(0).to(device)
            #             img_new_gpu = test_agent.model.decode(z_new_gpu)
            #             img_all = torch.cat((img_all, img_new_gpu), axis=0)
            #             image_pred = img_new_gpu.cpu().squeeze(0)

            # vutils.save_image(img_all[1:,...].cpu(),
            #             os.path.join(output_dir, model_type + '_traveler.png'),
            #             nrow=len(vars),
            #             normalize=True,
            #             range=(-1,1))
            # img_grid = vutils.make_grid(img_all[1:,...].cpu(), nrow=len(vars), normalize=True, range=(-1,1))
            # plt.imshow(img_grid.permute(1,2,0))
            # plt.axis('off')
            # plt.show()

            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            # ax1.plot(test_agent.iteration, test_agent.netE_losses, color='blue')
            # ax1.legend(['Encoder Loss'], loc='upper right')
            # ax2.plot(test_agent.iteration, test_agent.netG_losses, color='blue')
            # ax2.legend(['Generator Loss'], loc='upper right')
            # # ax2.set_ylabel('Loss')
            # ax3.plot(test_agent.iteration, test_agent.netD_losses, color='blue')
            # ax3.legend(['Discriminator Loss'], loc='upper right')
            # ax4.plot(test_agent.iteration, test_agent.kld_losses, color='blue')
            # ax4.legend(['KLD Loss'], loc='upper right')        
            # ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            # plt.xlabel('# of iter')
            # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        elif model_type in gan_model:
            model = gan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = GANTrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

            plot_train_losses(test_agent.get_train_history())
            print('Epoch = {:d}'.format(test_agent.get_current_epoch()))

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
        exit()

    # # 1) Linear Regression
    # if config['train_params']['train_reg'] :
    #     print('======= Linear Regression Controller =========')
    #     image_size = eval(config['train_params']['image_size'])
    #     preload_sample = config['train_params']['preload_sample']
    #     reg_num_prvs = config['train_params']['reg_num_prvs']
    #     reg_type = config['train_params']['reg_type']
    #     reg_weight_filename = config['train_params']['reg_weight_filename']
    #     reg_model_filename = config['train_params']['reg_model_filename']
    #     use_multicore = config['train_params']['use_multicore']

    #     reg_kwargs = {
    #         'dataset_dir': dataset_dir,
    #         'output_dir': output_dir,
    #         'image_size': image_size,
    #         'num_prvs': reg_num_prvs,
    #         'reg_type': reg_type,
    #         'weight_filename': reg_weight_filename,
    #         'model_filename': reg_model_filename,
    #         'preload': preload_sample,
    #         'printout': False,
    #         'subject_list': subject_list,
    #         'map_list': map_list,
    #         'iteration': 0,
    #     }

    #     # if use_multicore: 
    #     #     # Multi-processing training
    #     #     proc = mp.Process(target=RegTrain_multi, kwargs=reg_kwargs)
    #     #     proc.start()
    #     #     proc.join()
    #     # else:
    #     #     # Single-processing training
    #     #     RegTrain_single(**reg_kwargs)

    #     RegTrain_single_advanced(**reg_kwargs)

    # 2) VAE
    if config['train_params']['train_vae']:
        print('============== VAE training ================')

        # DataLoader
        print('Loading datasets from {:s}'.format(dataset_dir))
        if dataloader_type == 'simple':
            all_data = ImageDataset_simple(dataset_dir, resize=img_resize, preload=True, transform=transform_composed)
        elif dataloader_type == 'advanced':
            all_data = ImageDataset_advanced(dataset_dir, subject_list, map_list, iter=0, resize=img_resize, preload=True, transform=transform_composed)
        else:
            raise Exception("Unknown dataloader_type {:s}".format(dataloader_type))
        
        train_data, test_data = train_test_split(all_data,
                                            test_size=config['dataset_params']['test_size'],
                                            random_state=config['dataset_params']['manual_seed'])       
        print('Load datasets successfully!')

        # Set up the model
        model_type = config['train_params']['model_type']
        try:
            file = open(os.path.join(setup_path.parent_dir, 'configs', model_type + '.yaml'), 'r')
            model_config = yaml.safe_load(file)
            file.close()
        except Exception as error:
            print_msg(str(error), type=3)
            exit()

        # Create the agent
        if model_type in vae_model:
            model = vae_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            train_agent = VAETrain(model,
                                device,
                                is_eval=False,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

        elif model_type in vaegan_model:
            model = vaegan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            train_agent = VAEGANTrain(model,
                                device,
                                is_eval=False,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])
        
        elif model_type in gan_model:
            model = gan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            train_agent = GANTrain(model,
                                device,
                                is_eval=False,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

        else:
            raise Exception("Unknown model_type {:s}".format(model_type))     

        

        # Training loop
        print('\n*** Start training ***')
        train_agent.load_dataset(train_data, test_data)
        train_agent.train()
        print('Trained ' + model_type + ' model successfully.')

    # # # 3) Controller Network
    # # if config['train_params']['train_latent']:
    # #     print('============= Latent Controller ==============')
    # #     batch_size = config['train_params']['latent_batch_size']
    # #     n_epochs = config['train_params']['latent_n_epochs']
    # #     z_dim = config['train_params']['z_dim']
    # #     img_resize = eval(config['train_params']['img_resize'])
    # #     latent_checkpoint_filename = config['train_params']['latent_checkpoint_filename']
    # #     latent_checkpoint_preload = config['train_params']['latent_checkpoint_preload']
    # #     latent_model_filename = config['train_params']['latent_model_filename']
    # #     latent_num_prvs = config['train_params']['latent_num_prvs']
    # #     latent_learning_rate = config['train_params']['latent_learning_rate']
    # #     vae_model_filename = config['train_params']['vae_model_filename']
        
    # #     # DataLoader
    # #     print('Loading latent datasets...')
    # #     data = LatentDataset(dataset_dir, num_prvs=latent_num_prvs, resize=img_resize, preload=False)
    # #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6)
    # #     print('Load latent datasets successfully.')

    # #     # Create agent
    # #     latent_agent = LatentTrain(MyLatent(z_dim+latent_num_prvs+1), MyVAE(z_dim), latent_learning_rate)
    # #     latent_agent.load_VAEmodel(os.path.join(output_dir, vae_model_filename))
    # #     if latent_checkpoint_preload:
    # #         latent_agent.load_checkpoint(os.path.join(output_dir, latent_checkpoint_filename))

    # #     # Training loop
    # #     for epoch in range(1, n_epochs + 1):
    # #         latent_agent.train(epoch, data_loader)
    # #         # latent_agent.test(test_loader)

    # #         if epoch % 10 == 0 and epoch > 0:
    # #             latent_agent.save_checkpoint(epoch, os.path.join(output_dir, latent_checkpoint_filename))
    # #             latent_agent.save_model(os.path.join(output_dir, latent_model_filename))
    # #     print('Trained Latent model successfully.')   