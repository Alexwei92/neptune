import setup_path
import cv2
import time, datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml
import threading
import random

import airsim
from utils import *
from controller import *

if __name__ == '__main__':

    # Read YAML configurations
    try:
        file = open('config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    # Simulation settings
    loop_rate = config['sim_params']['loop_rate']
    output_path = config['sim_params']['output_path']
    save_data = config['sim_params']['save_data']
    agent_type = config['sim_params']['agent_type']
    dagger_type = config['sim_params']['dagger_type']
    train_mode = config['sim_params']['train_mode']
    initial_pose = eval(config['sim_params']['initial_pose'])

    # Control settings
    max_yawRate = config['ctrl_params']['max_yawRate']
    forward_speed = config['ctrl_params']['forward_speed']
    mission_height = config['ctrl_params']['mission_height']
    image_size = eval(config['ctrl_params']['image_size'])
    use_rangefinder = config['ctrl_params']['use_rangefinder']
    model_path = config['ctrl_params']['model_path']

    # Joystick/RC settings
    joy = Joystick(config['rc_params']['device_id']) 
    yaw_axis = config['rc_params']['yaw_axis']
    type_axis = config['rc_params']['type_axis']
    mode_axis = config['rc_params']['mode_axis']

    # Visualize settings
    plot_heading = config['visualize_params']['plot_heading']
    plot_cmd = config['visualize_params']['plot_cmd']
    plot_2Dpos = config['visualize_params']['plot_2Dpos']

    # Fast Loop Init
    kwargs = {
        'agent_type': agent_type,
        'dagger_type': dagger_type,
        'image_size': image_size,
        'initial_pose': initial_pose,
        'loop_rate': loop_rate,
        'max_yawRate': max_yawRate,
        'mission_height': mission_height,
        'forward_speed': forward_speed,
        'train_mode': train_mode,
        'use_rangefinder': use_rangefinder,
        'joystick': joy,
        'yaw_axis': yaw_axis,
        'type_axis': type_axis,
        'mode_axis': mode_axis,                    
    }
    fast_loop = FastLoop(**kwargs)

    # Control Agent Init
    if agent_type == 'reg':
        # Linear regression controller
        reg_num_prvs = config['train_params']['reg_num_prvs']
        reg_weight_path = os.path.join(setup_path.parent_dir, model_path, 'reg_weight_test.csv')
        controller_agent = RegCtrl(reg_num_prvs, image_size, reg_weight_path, printout=False)
    elif agent_type == 'latent':
        # Latent NN controller
        z_dim = config['train_params']['z_dim']
        img_resize = eval(config['train_params']['img_resize'])
        latent_num_prvs = config['train_params']['latent_num_prvs']
        vae_model_path = os.path.join(setup_path.parent_dir, model_path, 'vae_model_z_15.pt')
        latent_model_path = os.path.join(setup_path.parent_dir, model_path, 'latent_model_z_15.pt')
        controller_agent = LatentCtrl(vae_model_path, latent_model_path, z_dim, latent_num_prvs, img_resize)
    elif agent_type == 'none':
        # Manual control
        dagger_type = 'none'
        print_msg('No agent controller enabled.')
    else:
        print_msg('Unknow agent_type: ' + agent_type, type=3)
        exit()

    # Visualize Init
    disp_handle = Display(image_size, max_yawRate, plot_heading, plot_cmd) 

    if plot_2Dpos:
        fig, ax = plt.subplots()
        pos_handle = DynamicPlot(fig, ax, max_width=120*loop_rate)

    # Data logger Init
    if save_data:
        data_logger = Logger(os.path.join(setup_path.parent_dir, output_path))

    # Reset function
    def reset():
        if fast_loop.agent_type == 'reg':
            controller_agent.reset_prvs()
        if fast_loop.agent_type == 'latent':
            controller_agent.reset_prvs()
        if save_data:
            data_logger.reset_folder(crashed=True)
        if plot_2Dpos:
            pos_handle.reset()

    '''
    Main Code
    '''
    try:

        # Multi-threading process for display
        disp_thread = threading.Thread(target=disp_handle.run)
        disp_thread.start()

        # Multi-threading process for state machine
        fastloop_thread = threading.Thread(target=fast_loop.run)
        fastloop_thread.start()

        time.sleep(1.0)
        print_msg("Ready to fly!", type=1)
        while True:
            start_time = time.perf_counter() # loop start time
                      
            # check collision
            if fast_loop.trigger_reset:
                reset()
                fast_loop.trigger_reset = False

            # Data logging
            if save_data:
                if fast_loop.manual_stop:
                    data_logger.reset_folder(crashed=False)
                    fast_loop.manual_stop = False

                if fast_loop.flight_mode == 'mission':
                    if (dagger_type == 'hg') and (not fast_loop.is_expert):
                        # in Hg-dagger, only log data in manual mode
                        pass
                    else:
                        data_logger.save_image('color', fast_loop.image_color)
                        data_logger.save_image('depth', fast_loop.image_depth)
                        data_logger.save_csv(fast_loop.drone_state.timestamp, fast_loop.drone_state.kinematics_estimated, fast_loop.pilot_cmd)
            
            # for regression controller:
            if fast_loop.is_expert and agent_type == 'reg':
                controller_agent.reset_prvs()

            # Update agent controller command
            if (not fast_loop.is_expert) and (fast_loop.flight_mode == 'mission'):
                if agent_type == 'reg':
                    fast_loop.agent_cmd = controller_agent.predict(fast_loop.image_color, fast_loop.image_depth, fast_loop.get_yaw_rate())
                elif agent_type == 'latent':
                    fast_loop.agent_cmd = controller_agent.predict(fast_loop.image_color, fast_loop.get_yaw_rate())
                else:
                    raise Exception('You must define an agent type!')

            # Update plots
            if train_mode == 'test':
                if fast_loop.is_expert:
                    disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, True)
                else:
                    disp_handle.update(fast_loop.image_color, fast_loop.agent_cmd, False)
            elif train_mode == 'train':
                if dagger_type == 'vanilla' or dagger_type == 'none':
                    disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, True)
                elif dagger_type == 'hg':
                    if fast_loop.is_expert:
                        disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, True)
                    else:
                        disp_handle.update(fast_loop.image_color, fast_loop.agent_cmd, False)
                else:
                    raise Exception('Unknown Dagger type: ' + dagger_type)
            else:
                raise Exception('Unknown train_mode: ' + train_mode)

            if plot_2Dpos:
                pos_handle.update(round(fast_loop.drone_state.kinematics_estimated.position.x_val, 3), 
                                  round(fast_loop.drone_state.kinematics_estimated.position.y_val, 3))

            # Ensure that the loop is running at a fixed rate
            elapsed_time = time.perf_counter() - start_time
            if (1./loop_rate - elapsed_time) < 0.0:
                print_msg('The main loop rate {:.2f} Hz is below {:.2f} Hz, consider to reduce the rate!'.format(1./elapsed_time, loop_rate), type=2)
            else:
                time.sleep(1./loop_rate - elapsed_time)

            # for CV plotting
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                break

            if (key == ord('k')):
                fast_loop.force_reset = True

    except Exception as error:
        print_msg(str(error), type=3)

    finally:
        print('===============================')
        print('Clean up the code...')
        disp_handle.is_active = False
        fast_loop.is_active = False
        joy.clean()
        if save_data:
            data_logger.clean()
        print('Exit the program successfully!')