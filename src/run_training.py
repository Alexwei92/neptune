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

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
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
    iteration = config['dataset_params']['iteration']
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
            print(str(error), type=3)
            exit()

        print('Model Type: {:s}'.format(model_type))
        if model_type in vae_model:
            model = vae_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = VAETrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

        elif model_type in vaegan_model:
            model = vaegan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = VAEGANTrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

        elif model_type in gan_model:
            model = gan_model[model_type](**model_config['model_params'])
            model_config['log_params']['output_dir'] = output_dir
            test_agent = GANTrain(model,
                                device,
                                is_eval=True,
                                train_params=model_config['train_params'],
                                log_params=model_config['log_params'])

        else:
            raise Exception("Unknown model_type {:s}".format(model_type))

        # training loss
        plot_train_losses(test_agent.get_train_history(), save_path=os.path.join(output_dir, model_type, 'train_loss.png'))
        print('Epoch = {:d}'.format(test_agent.get_current_epoch()))
        
        if model_type not in gan_model:
            # KLD loss
            plot_KLD_losses(test_agent.get_train_history(), plot_sum=False, plot_mean=True,
                        save_path=os.path.join(output_dir, model_type, 'kld_loss_z.png'))
        
            # latent traversal
            if dataloader_type == 'simple':
                all_data = ImageDataset_simple(dataset_dir, resize=img_resize, transform=transform_composed)
            elif dataloader_type == 'advanced':
                all_data = ImageDataset_advanced(dataset_dir, subject_list, map_list, iter=0, resize=img_resize, transform=transform_composed)
            else:
                raise Exception("Unknown dataloader_type {:s}".format(dataloader_type))
            
            _, test_data = train_test_split(all_data,
                                        test_size=config['dataset_params']['test_size'],
                                        random_state=config['dataset_params']['manual_seed'])
            
            traversal = np.linspace(-3.0, 3.0, num=10)
            img_all = test_agent.latent_traversal(test_data[50], traversal)
            vutils.save_image(img_all[1:,...].cpu(),
                        os.path.join(output_dir, model_type, 'latent_traversal.png'),
                        nrow=len(traversal),
                        normalize=True,
                        range=(-1,1))
            img_grid = vutils.make_grid(img_all[1:,...].cpu(),
                        nrow=len(traversal),
                        normalize=True,
                        range=(-1,1))
            
            plt.figure()
            plt.imshow(img_grid.permute(1,2,0))
            plt.axis('off')


        # print('\n***** NN Controller Results *****')
        # latent_checkpoint_filename = config['train_params']['latent_checkpoint_filename']
        # latent_num_prvs = config['train_params']['latent_num_prvs']
        # latent_agent = LatentTrain(MyLatent(z_dim+latent_num_prvs+1), MyVAE(z_dim))
        # latent_agent.load_checkpoint(os.path.join(output_dir, latent_checkpoint_filename))
        # latent_agent.plot_train_result()
        # print('****************************\n')

        plt.show()
        exit()

    # 1) Linear Regression
    if config['train_params']['train_reg'] :
        print('======= Linear Regression Controller =========')
        image_size = eval(config['train_params']['image_size'])
        preload_sample = config['train_params']['preload_sample']
        reg_num_prvs = config['train_params']['reg_num_prvs']
        reg_prvs_mode = config['train_params']['reg_prvs_mode']
        reg_type = config['train_params']['reg_type']
        reg_weight_filename = config['train_params']['reg_weight_filename']
        reg_model_filename = config['train_params']['reg_model_filename']
        use_multicore = config['train_params']['use_multicore']

        reg_kwargs = {
            'dataset_dir': dataset_dir,
            'output_dir': output_dir,
            'image_size': image_size,
            'num_prvs': reg_num_prvs,
            'prvs_mode': reg_prvs_mode,
            'reg_type': reg_type,
            'weight_filename': reg_weight_filename,
            'model_filename': reg_model_filename,
            'preload': preload_sample,
            'printout': False,
        }

        if dataloader_type == 'advanced':
            for subject in subject_list:
                reg_kwargs.update({
                    'output_dir': os.path.join(output_dir, subject, 'iter'+str(iteration), 'reg'),
                    'subject_list': subject,
                    'map_list': map_list,
                    'iteration': iteration,
                    })
                RegTrain_multi_advanced(**reg_kwargs)

        elif dataloader_type == 'simple':
            if use_multicore: 
                # Multi-processing training
                proc = mp.Process(target=RegTrain_multi, kwargs=reg_kwargs)
                proc.start()
                proc.join()
            else:
                # Single-processing training
                RegTrain_single(**reg_kwargs)

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

        # Random seed
        torch.manual_seed(model_config['train_params']['manual_seed'])
        torch.cuda.manual_seed(model_config['train_params']['manual_seed'])
        np.random.seed(model_config['train_params']['manual_seed'])

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

    # 3) Controller Network
    if config['train_params']['train_latent']:
        print('============= Latent Controller ==============')
        img_resize = eval(config['train_params']['img_resize'])
        vae_model_type = config['train_params']['vae_model_type']
        vae_model_path = os.path.join(folder_path, config['train_params']['vae_model_path'])
        latent_model_type = config['train_params']['latent_model_type']

        # Latent model config
        try:
            file = open(os.path.join(setup_path.parent_dir, 'configs', latent_model_type + '.yaml'), 'r')
            latent_model_config = yaml.safe_load(file)
            file.close()
        except Exception as error:
            print_msg(str(error), type=3)
            exit()
        
        num_prvs = latent_model_config['model_params']['num_prvs']
        prvs_mode = latent_model_config['model_params']['prvs_mode']
        torch.manual_seed(latent_model_config['train_params']['manual_seed'])
        torch.cuda.manual_seed(latent_model_config['train_params']['manual_seed'])
        np.random.seed(latent_model_config['train_params']['manual_seed'])

        # VAE model config
        try:
            file = open(os.path.join(setup_path.parent_dir, 'configs', vae_model_type + '.yaml'), 'r')
            vae_model_config = yaml.safe_load(file)
            file.close()
        except Exception as error:
            print_msg(str(error), type=3)
            exit()
        
        # DataLoader
        if dataloader_type == 'simple':
            

            all_data = LatentDataset_simple(dataset_dir,
                                    num_prvs=num_prvs,
                                    prvs_mode=prvs_mode,
                                    resize=img_resize,
                                    transform=transform_composed)
            train_data, test_data = train_test_split(all_data,
                                                test_size=config['dataset_params']['test_size'],
                                                random_state=config['dataset_params']['manual_seed'])     
            print('Load latent datasets successfully.')    

            # Create the agent
            latent_model = latent_model[latent_model_type](**latent_model_config['model_params'])
            vae_model = vae_model[vae_model_type](**vae_model_config['model_params'])
            latent_model_config['log_params']['output_dir'] = output_dir
            train_agent = LatentTrain(latent_model,
                                vae_model,
                                vae_model_path,
                                device,
                                is_eval=False,
                                train_params=latent_model_config['train_params'],
                                log_params=latent_model_config['log_params'])
                
            print('\n*** Start training ***')
            train_agent.load_dataset(train_data, test_data)
            train_agent.train()
            print('Trained Latent model successfully.')   

        elif dataloader_type == 'advanced':
            for subject in subject_list:
                all_data = LatentDataset_advanced(dataset_dir,
                                    subject_list=[subject],
                                    map_list=map_list,
                                    iter=iteration,
                                    num_prvs=num_prvs,
                                    prvs_mode=prvs_mode,
                                    resize=img_resize,
                                    transform=transform_composed)

                train_data, test_data = train_test_split(all_data,
                                                    test_size=config['dataset_params']['test_size'],
                                                    random_state=config['dataset_params']['manual_seed'])     
                print('Load latent datasets successfully.')
                
                # Create the agent
                latent_model = latent_model[latent_model_type](**latent_model_config['model_params'])
                if vae_model_type in vae_model:
                    vae_model = vae_model[vae_model_type](**vae_model_config['model_params'])
                elif vae_model_type in vaegan_model:
                    vae_model = vaegan_model[vae_model_type](**vae_model_config['model_params'])
                latent_model_config['log_params']['output_dir'] = os.path.join(output_dir, subject, 'iter'+str(iteration))
                train_agent = LatentTrain(latent_model,
                                    vae_model,
                                    vae_model_path,
                                    device,
                                    is_eval=False,
                                    train_params=latent_model_config['train_params'],
                                    log_params=latent_model_config['log_params'])
                    
                print('\n*** Start training ***')
                train_agent.load_dataset(train_data, test_data)
                train_agent.train()
                print('Trained Latent model successfully.')   