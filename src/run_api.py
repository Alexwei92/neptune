import setup_path
import cv2
import time, datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml
import threading

import airsim
from utils import *
from controller import *

if __name__ == '__main__':
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Read YAML configurations
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            raise Exception

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
    plot_2Dpos = config['visualize_params']['plot_2Dpos']


    # Visualize Init
    disp_handle = Display(
        image_size=image_size, 
        loop_rate=loop_rate,
        plot_heading=plot_heading)
    
    if plot_2Dpos:
        fig, ax = plt.subplots()
        pos_handle = DynamicPlot(fig, ax, max_width=120*loop_rate)

    # Controller Init
    controller = Controller(client=client,
                            forward_speed=forward_speed, 
                            height=height, 
                            max_yawRate=max_yawRate)

    if agent_type == 'reg':
        # Linear regression controller
        reg_weight_path = os.path.join(setup_path.parent_dir, model_path, 'reg_weight.csv')
        controller_agent = RegCtrl(
                            image_size=image_size,
                            weight_file_path=reg_weight_path)

    elif agent_type == 'latent':
        # latent controller
        z_dim = config['train_params']['z_dim']
        img_resize = eval(config['train_params']['img_resize'])
        vae_model_path = os.path.join(setup_path.parent_dir, model_path, 'vae_model.pt')
        latent_model_path = os.path.join(setup_path.parent_dir, model_path, 'latent_model.pt')
        controller_agent = LatentCtrl(
                            vae_model_path=vae_model_path,
                            latent_model_path=latent_model_path,
                            z_dim=z_dim,
                            image_resize=img_resize)
    else:
        dagger_type = 'none'
        print('No agent controller currently available.')


    # State Machine Init
    state_machine = StateMachine(
        agent_type=agent_type,
        dagger_type=dagger_type,
        train_mode=train_mode,
    )

    # Data logger Init
    data_logger = Logger(root_dir=os.path.join(setup_path.parent_dir, output_path),
                        save_data=save_data)

    '''
    Main Code
    '''
    try:
        # Enable API control
        client.enableApiControl(True)
        client.armDisarm(True)

        # Initial pose
        # init_position = airsim.Vector3r(115, 181, -height)
        # init_orientation = airsim.to_quaternion(0, 0, np.pi*np.random.rand())
        # intial_pose = airsim.Pose(init_position, init_orientation)

        init_position = airsim.Vector3r(initial_pose[0], initial_pose[1], -initial_pose[2])
        init_orientation = airsim.to_quaternion(0, 0, initial_pose[3])
        client.simSetVehiclePose(airsim.Pose(init_position, init_orientation), True)
        time.sleep(0.5)

        # Take-off
        print("Taking off, please wait...")
        client.takeoffAsync().join()
        print("Start the loop")

        # Multithreading process for display
        disp_thread = threading.Thread(target=disp_handle.run)
        disp_thread.start()

        while True:
            now = time.time() # loop start time

            # Check collision
            state_machine.check_collision(client.simGetCollisionInfo().has_collided)
            if client.simGetCollisionInfo().has_collided:
                client.reset()
                client.enableApiControl(True)
                client.simSetVehiclePose(generate_random_pose(height), True)
                if save_data:
                    data_logger.crash_count += 1

            # Get Multirotor estimated states
            drone_state = client.getMultirotorState()
            
            # Update yaw command from RC/joystick
            yaw_cmd = joy.get_input(yaw_axis)

            # Update flight mode from RC/joystick
            state_machine.set_flight_mode(joy.get_input(mode_axis))

            # Update controller type from RC/joystick
            state_machine.set_controller_type(joy.get_input(type_axis))

            # Get the Images in FPV
            camera_response = client.simGetImages([
                # uncompressed RGB array bytes 
                airsim.ImageRequest('0', airsim.ImageType.Scene, False, False),
                # floating point uncompressed image
                airsim.ImageRequest('1', airsim.ImageType.DepthVis, True, False)])

            for _, response in enumerate(camera_response):
                # Color image
                if response.image_type == airsim.ImageType.Scene:
                    image_color = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    image_color = image_color.reshape(image_size[0], image_size[1], 3)
                # Depth image
                if response.image_type == airsim.ImageType.DepthVis:
                    image_depth = np.asarray(response.image_data_float, dtype=np.float32)
                    image_depth = image_depth.reshape(image_size[0], image_size[1])
                    image_depth = (image_depth * 255).astype(np.uint8)

            # Data logging
            if state_machine.get_flight_mode() == 'mission' and save_data:
                data_logger.save_image('color', image_color)
                data_logger.save_image('depth', image_depth)
                data_logger.save_csv(drone_state.timestamp, drone_state.kinematics_estimated, yaw_cmd)

            # Update controller commands
            controller.set_current_yaw(get_yaw_from_orientation(drone_state.kinematics_estimated.orientation))
            if state_machine.is_expert or state_machine.flight_mode == 'hover':
                controller.step(yaw_cmd, state_machine.get_flight_mode())
            
            else:
                if state_machine.agent_type == 'reg':
                    yawRate = drone_state.kinematics_estimated.angular_velocity.z_val
                    yaw_cmd = controller_agent.predict(image_color, image_depth, yawRate)
                    controller.step(yaw_cmd, 'mission')

                elif state_machine.agent_type == 'latent':
                    yaw_cmd = controller_agent.predict(image_color)
                    controller.step(yaw_cmd, 'mission')
                
                else:
                    pass

            # Update plots
            disp_handle.update(image_color, yaw_cmd, state_machine.is_expert)

            if plot_2Dpos:
                pos_handle.update(drone_state.kinematics_estimated.position.x_val, drone_state.kinematics_estimated.position.y_val)

            # for CV plotting
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                client.reset()
                break

            # Ensure that the loop is running at a fixed rate
            elapsed_time = time.time() - now
            if (1./loop_rate - elapsed_time) < 0.0:
                print('[WARNING] The main loop rate is too high, consider to reduce the rate!')
                print('Real-time loop rate: {:.2f}'.format(1./elapsed_time))
            else:
                time.sleep(1./loop_rate - elapsed_time)

    except Exception as e:
        print(e)

    finally:
        print('===============================')
        print('Clean up the code...')
        disp_handle.is_active = False
        disp_handle.clean()
        joy.clean()
        data_logger.clean()
        client.armDisarm(False)
        client.enableApiControl(False)
        client.reset()
        print('Exit the program successfully!')