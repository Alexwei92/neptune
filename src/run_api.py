import setup_path
import cv2
import time, datetime
import os
import numpy as np
import yaml
import threading

import airsim
from utils import *
from controller import *

if __name__ == '__main__':

    # Read YAML configurations
    try:
        file = open('api_config.yaml', 'r')
        config = yaml.safe_load(file)
        file.close()
    except Exception as error:
        print_msg(str(error), type=3)
        exit()

    # Simulation settings
    loop_rate = config['sim_params']['loop_rate']
    output_dir = config['sim_params']['output_dir']
    folder_path = config['sim_params']['folder_path']
    if len(folder_path) == 0: # if leave it blank
        output_dir = os.path.join(setup_path.parent_dir, output_dir)
    else:
        output_dir = os.path.join(folder_path, output_dir)
    save_data = config['sim_params']['save_data']
    agent_type = config['sim_params']['agent_type']
    dagger_type = config['sim_params']['dagger_type']
    train_mode = config['sim_params']['train_mode']
    initial_pose = eval(config['sim_params']['initial_pose'])
    if not isinstance(initial_pose[0], tuple):
        initial_pose = (initial_pose, initial_pose) # to use random.choice properly

    # Control settings
    max_yawRate = config['ctrl_params']['max_yawRate']
    forward_speed = config['ctrl_params']['forward_speed']
    mission_height = config['ctrl_params']['mission_height']
    image_size = eval(config['ctrl_params']['image_size'])
    use_rangefinder = config['ctrl_params']['use_rangefinder']
    
    # Joystick/RC settings
    joy = Joystick(config['rc_params']['device_id']) 
    yaw_axis = config['rc_params']['yaw_axis']
    type_axis = config['rc_params']['type_axis']
    mode_axis = config['rc_params']['mode_axis']

    # Visualize settings
    plot_heading = config['visualize_params']['plot_heading']
    plot_cmd = config['visualize_params']['plot_cmd']
    plot_trajectory = config['visualize_params']['plot_trajectory']

    # Fast Loop Init
    API_kwargs = {
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
    fast_loop = FastLoop(**API_kwargs)

    # Control Agent Init
    if agent_type == 'reg':
        # Linear regression controller
        num_prvs = config['agent_params']['reg_num_prvs']
        prvs_mode = config['agent_params']['reg_prvs_mode']
        model_path = config['agent_params']['reg_model_path']
        model_path = os.path.join(folder_path, model_path)
        controller_agent = RegCtrl(num_prvs, prvs_mode, image_size, model_path, printout=False)
    elif agent_type == 'latent':
        # Latent NN controller
        img_resize = (64, 64)
        vae_model_path = os.path.join(folder_path, config['agent_params']['vae_model_path'])
        vae_model_type = config['agent_params']['vae_model_type']
        latent_model_path = os.path.join(folder_path, config['agent_params']['latent_model_path'])
        latent_model_type = config['agent_params']['latent_model_type']
        controller_agent = LatentCtrl(vae_model_path,
                                vae_model_type,
                                latent_model_path,
                                latent_model_type,
                                img_resize)
    elif agent_type == 'none':
        # Manual control
        dagger_type = 'none'
        print_msg('Agent controller disabled.')
    else:
        print_msg('Unknow agent_type: ' + agent_type, type=3)
        exit()

    # Visualize Init
    disp_handle = Display(loop_rate, image_size, max_yawRate, plot_heading, plot_cmd, plot_trajectory) 

    # Data logger Init
    if save_data:
        data_logger = Logger(output_dir)

    # Reset function
    def reset():
        if fast_loop.agent_type is not 'none':
            fast_loop.controller.reset_cmd_history()
        if save_data:
            data_logger.reset_folder('crashed')
        if plot_trajectory:
            disp_handle.trajectory_handle.reset()

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
                      
            # If reset
            if fast_loop.trigger_reset:
                reset()
                fast_loop.trigger_reset = False

            # Data logging
            if save_data:
                if fast_loop.manual_stop:
                    # when the user manually stop the mission
                    data_logger.reset_folder('safe')
                    fast_loop.manual_stop = False

                # if fast_loop.virtual_crash:
                #     # when the pilot think it may crash
                #     data_logger.reset_folder('virtual_crash')
                #     fast_loop.virtual_crash = False

                if fast_loop.flight_mode == 'mission':
                    data_logger.save_image('color', fast_loop.image_color)
                    data_logger.save_image('depth', fast_loop.image_depth)
                    data_logger.update_flag(fast_loop.is_expert)
                    data_logger.save_csv(fast_loop.drone_state.timestamp, fast_loop.drone_state.kinematics_estimated, fast_loop.pilot_cmd)

            # Update agent controller command
            if (not fast_loop.is_expert) and (fast_loop.flight_mode == 'mission'):
                if agent_type == 'reg':
                    fast_loop.agent_cmd = controller_agent.predict(fast_loop.image_color, fast_loop.image_depth, \
                                                                fast_loop.get_yaw_rate(), fast_loop.controller.cmd_history)
                elif agent_type == 'latent':
                    fast_loop.agent_cmd = controller_agent.predict(fast_loop.image_color, fast_loop.get_yaw_rate(), \
                                                                fast_loop.controller.cmd_history)
                else:
                    raise Exception('You must define an agent controller type!')

            # Update plots
            if train_mode == 'test':
                if fast_loop.is_expert:
                    disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, is_expert=True)
                else:
                    disp_handle.update(fast_loop.image_color, fast_loop.agent_cmd, is_expert=False)
            elif train_mode == 'train':
                if dagger_type == 'vanilla' or dagger_type == 'none':
                    disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, is_expert=True)
                elif dagger_type == 'hg':
                    if fast_loop.is_expert:
                        disp_handle.update(fast_loop.image_color, fast_loop.pilot_cmd, is_expert=True)
                    else:
                        disp_handle.update(fast_loop.image_color, fast_loop.agent_cmd, is_expert=False)
                else:
                    raise Exception('Unknown dagger_type: ' + dagger_type)
            else:
                raise Exception('Unknown train_mode: ' + train_mode)

            if plot_trajectory:
                disp_handle.update_trajctory(fast_loop.drone_state.kinematics_estimated.position.x_val,
                                             fast_loop.drone_state.kinematics_estimated.position.y_val)

            # Ensure that the loop is running at a fixed rate
            elapsed_time = time.perf_counter() - start_time
            if (1./loop_rate - elapsed_time) < 0.0:
                print_msg('The controller loop is running at {:.2f} Hz, expected {:.2f} Hz!'.format(1./elapsed_time, loop_rate), type=2)
            else:
                time.sleep(1./loop_rate - elapsed_time)

            # For CV plot
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                break

            # Manual reset
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