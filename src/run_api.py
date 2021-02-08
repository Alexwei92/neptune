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
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

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
    height = config['ctrl_params']['height']
    image_size = eval(config['ctrl_params']['image_size'])
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

    # Controller Init
    controller = Controller(client, forward_speed, height, max_yawRate)

    if agent_type == 'reg':
        # Linear regression controller
        reg_weight_path = os.path.join(setup_path.parent_dir, model_path, 'reg_weight.csv')
        controller_agent = RegCtrl(image_size, reg_weight_path, True)

    elif agent_type == 'latent':
        # Latent NN controller
        z_dim = config['train_params']['z_dim']
        img_resize = eval(config['train_params']['img_resize'])
        vae_model_path = os.path.join(setup_path.parent_dir, model_path, 'vae_model.pt')
        latent_model_path = os.path.join(setup_path.parent_dir, model_path, 'latent_model.pt')
        controller_agent = LatentCtrl(
                            vae_model_path=vae_model_path,
                            latent_model_path=latent_model_path,
                            z_dim=z_dim,
                            image_resize=img_resize)
    
    elif agent_type == 'none':
        # manual control
        dagger_type = 'none'
        print_msg('No agent controller enabled.')
    
    else:
        print_msg('Unknow agent_type: ' + agent_type, type=3)
        exit()

    # State Machine Init
    state_machine = StateMachine(agent_type, train_mode)

    # Rangfinder Init
    rangefinder = Rangefinder()

    # Visualize Init
    disp_handle = Display(image_size, max_yawRate, loop_rate, plot_heading, plot_cmd) 

    if plot_2Dpos:
        fig, ax = plt.subplots()
        pos_handle = DynamicPlot(fig, ax, max_width=120*loop_rate)

    # Data logger Init
    if save_data:
        data_logger = Logger(os.path.join(setup_path.parent_dir, output_path))

    # Reset function
    def reset():
        reset_environment(client, state_machine, random.choice(initial_pose))
        rangefinder.reset()
        if state_machine.agent_type == 'reg':
            controller_agent.reset_prvs()
        if save_data:
            data_logger.reset_folder()
        if plot_2Dpos:
            pos_handle.reset()

    '''
    Main Code
    '''
    try:
        # Enable API control
        client.enableApiControl(True)
        client.armDisarm(True)

        # Take-off
        client.takeoffAsync()

        # Initial pose
        client.simSetVehiclePose(add_offset_to_pose(random.choice(initial_pose)), ignore_collison=True)
        time.sleep(0.5)
        
        # Multi-threading process for display
        disp_thread = threading.Thread(target=disp_handle.run)
        disp_thread.start()

        print_msg("Ready to fly!", type=1)
        while True:
            start_time = time.perf_counter() # loop start time

            # Check collision
            if client.simGetCollisionInfo().has_collided:
                reset()

            # Get Multirotor estimated states
            drone_state = client.getMultirotorState()

            # Update rangefinder height
            rangefinder_distance = client.getDistanceSensorData().distance
            rangefinder.update(rangefinder_distance)

            # Update pilot yaw command from RC/joystick
            pilot_cmd = joy.get_input(yaw_axis)
            pilot_cmd = round(pilot_cmd, PRECISION)  
        
            # Update flight mode from RC/joystick
            state_machine.set_flight_mode(joy.get_input(mode_axis))

            # Update controller type from RC/joystick
            state_machine.set_controller_type(joy.get_input(type_axis))

            # Get images
            camera_response = client.simGetImages([
                # uncompressed RGB array bytes 
                airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
                # floating point uncompressed image
                airsim.ImageRequest('1', airsim.ImageType.DepthVis, pixels_as_float=True, compress=False)])
            image_color, image_depth = get_camera_images(camera_response, image_size)

            # for regression controller:
            if state_machine.is_expert and agent_type == 'reg':
                controller_agent.reset_prvs()

            # Data logging
            if save_data and state_machine.get_flight_mode() == 'mission':
                if (dagger_type == 'hg') and (not state_machine.is_expert):
                    # in Hg-dagger, only log data in manual mode
                    pass
                else:
                    data_logger.save_image('color', image_color)
                    data_logger.save_image('depth', image_depth)
                    data_logger.save_csv(drone_state.timestamp, drone_state.kinematics_estimated, pilot_cmd)

            # Update controller commands
            current_yaw = get_yaw_from_orientation(drone_state.kinematics_estimated.orientation)
            controller.set_current_yaw(current_yaw)

            if state_machine.is_expert or state_machine.flight_mode == 'hover':
                controller.step(pilot_cmd, rangefinder.get_filtered_height(), state_machine.get_flight_mode())
                agent_cmd = pilot_cmd
            else:
                if state_machine.agent_type == 'reg':
                    yawRate = drone_state.kinematics_estimated.angular_velocity.z_val
                    agent_cmd = controller_agent.predict(image_color, image_depth, yawRate)
                    controller.step(agent_cmd, rangefinder.get_filtered_height(), 'mission')
                elif state_machine.agent_type == 'latent':
                    agent_cmd = controller_agent.predict(image_color)
                    controller.step(agent_cmd, rangefinder.get_filtered_height(), 'mission')
                else:
                    raise Exception('You must define an agent type!')

            # Update plots
            if train_mode == 'test':
                if state_machine.is_expert:
                    disp_handle.update(image_color, pilot_cmd, True)
                else:
                    disp_handle.update(image_color, agent_cmd, False)
            elif train_mode == 'train':
                if dagger_type == 'vanilla' or dagger_type =='none':
                    disp_handle.update(image_color, pilot_cmd, True)

                elif dagger_type == 'hg':
                    if state_machine.is_expert:
                        disp_handle.update(image_color, pilot_cmd, True)
                    else:
                        disp_handle.update(image_color, agent_cmd, False)
                else:
                    raise Exception('Unknown Dagger type: ' + dagger_type)
            else:
                raise Exception('Unknown train_mode: ' + train_mode)

            if plot_2Dpos:
                pos_handle.update(round(drone_state.kinematics_estimated.position.x_val, 3), 
                                  round(drone_state.kinematics_estimated.position.y_val, 3))

            # for CV plotting
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                break
            
            # Manual reset
            if (key==ord('k')):
                reset()

            # Ensure that the loop is running at a fixed rate
            elapsed_time = time.perf_counter() - start_time
            if (1./loop_rate - elapsed_time) < 0.0:
                print_msg('The main loop rate {:.2f} Hz is below {:.2f} Hz, consider to reduce the rate!'.format(1./elapsed_time, loop_rate), type=2)
            else:
                time.sleep(1./loop_rate - elapsed_time)

    except Exception as error:
        print_msg(str(error), type=3)

    finally:
        print('===============================')
        print('Clean up the code...')
        disp_handle.is_active = False
        disp_handle.clean()
        joy.clean()
        if save_data:
            data_logger.clean()
        client.armDisarm(False)
        client.enableApiControl(False)
        client.reset()
        print('Exit the program successfully!')