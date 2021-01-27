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

# Generate random pose
def generate_random_pose(height):
    position = airsim.Vector3r(2.5*np.random.rand(), 2.5*np.random.rand(), -height)
    orientation = airsim.to_quaternion(0, 0, np.pi*np.random.rand())
    intial_pose = airsim.Pose(position, orientation)
    return intial_pose

# Get yaw from orientation
def get_yaw_from_orientation(orientation):
    pitch, roll, yaw = airsim.to_eularian_angles(orientation) # in radians
    return yaw 

class StateMachine():
    '''
    A high-level state machine
    '''
    # Init
    def __init__(self, controller_type='expert', train_mode='train'):
        self.flight_mode = 'idle' # {idle, hover, mission}
        self.controller_type = controller_type 
        self.train_mode = train_mode
        self.has_collided = False

    # Collision
    def check_collision(self, has_collided):
        if has_collided:
            print('Collision occurred! Reset to random initial state.')
            self.has_collided = True
            self.flight_mode = 'hover' # Hover by default

    # Flight Mode
    def set_flight_mode(self, flight_mode):
        if self.has_collided and flight_mode=='mission':
            # force to start from hover when reset
            print('Please reset the stick to its idle position first!')
        else:
            self.flight_mode = flight_mode
            self.has_collided = False

    def get_flight_mode(self):
        return self.flight_mode

    # Controller
    def get_controller_type(self):
        return self.controller_type

class Logger():
    '''
    Data logger
    '''
    def __init__(self, root_dir, save_data=True):
        if save_data:
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)
            folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")
            path = root_dir+'/'+folder_name
            os.makedirs(path+'/'+'color')
            os.makedirs(path+'/'+'depth')
            self.path = path

            self.f = open(path+'/'+'airsim.csv', 'w', newline='')
            self.filewriter = csv.writer(self.f, delimiter = ',')
            self.filewriter.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'yaw_rate', 'yaw_cmd', 'crash_count', 'image_name'])
            self.crash_count = 0
            self.index = 0

    def save_image(self, type, img):
        cv2.imwrite(os.path.join(self.path+'/'+type, "%07i.png" % self.index), img)

    def save_csv(self, timestamp, state, yaw_cmd):
        values = [
            str(timestamp), # timestamp
            state.position.x_val, # pos_x
            state.position.y_val, # pos_y
            state.position.z_val, # pos_z
            get_yaw_from_orientation(state.orientation), # yaw
            state.angular_velocity.z_val, # yaw_rate
            yaw_cmd, # yaw_cmd
            self.crash_count, # crash_count
            self.index] # image_name
        self.filewriter.writerow(values)
        self.index += 1

    def clean(self):
        if hasattr(self, 'f'):
            self.f.close()

class Display():
    '''
    For displaying on the window
    '''
    def __init__(self, image_size, loop_rate=15, plot_heading=False):
        self.is_active = True
        self.image = np.zeros((image_size[0], image_size[1], 4))
        self.bar = 0.0
        self.loop_rate = loop_rate
        self.plot_heading = plot_heading
        self.win_name = 'disp'
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def update_bar(self, bar):
        self.bar = bar

    def update_image(self, image):
        self.image = image

    def update(self, image, bar):
        self.update_image(image)
        self.update_bar(bar)

    def run(self):
        while self.is_active:
            start_time = time.time()
            if self.plot_heading:
                plot_with_heading(self.win_name, self.image, self.bar/2.0)
            else:
                plot_without_heading(self.win_name, self.image)

            elapsed_time = time.time() - start_time
            if (1./self.loop_rate - elapsed_time) < 0.0:
                print('[WARNING] The visualize loop rate is too high, consider to reduce the rate!')
            else:
                time.sleep(1./self.loop_rate - elapsed_time)
        
        self.clean()
    
    def clean(self):
        cv2.destroyAllWindows()

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
    controller_type = config['sim_params']['controller_type']
    train_mode = config['sim_params']['train_mode']
    output_path = config['sim_params']['output_path']
    save_data = config['sim_params']['save_data']
    initial_pose = eval(config['sim_params']['initial_pose'])

    # Control settings
    max_yawRate = config['ctrl_params']['max_yawRate']
    forward_speed = config['ctrl_params']['forward_speed']
    height = config['ctrl_params']['height']
    image_size = eval(config['ctrl_params']['image_size'])

    # Joystick/RC settings
    joy = Joystick(config['rc_params']['device_id']) 
    yaw_axis = config['rc_params']['yaw_axis']
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

    # State Machine Init
    state_machine = StateMachine(
        controller_type=controller_type,
        train_mode=train_mode,
    )

    # Controller Init
    controller = ExpertCtrl(client=client,
                            forward_speed=forward_speed, 
                            height=height, 
                            max_yawRate=max_yawRate)

    # Data logger Init
    data_logger = Logger(root_dir=os.path.join(setup_path.parent_dir, output_path), save_data=save_data)

    '''
    Main Code
    '''
    try:
        # Enable API control
        client.enableApiControl(True)
        client.armDisarm(True)

        # # Initial pose
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
        state_machine.set_flight_mode('hover')
        controller.is_active = True
        print("Start the loop")

        # Multi-threading for display
        disp_thread = threading.Thread(target=disp_handle.run)
        disp_thread.start()

        # Main loop
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
            mode_cmd = joy.get_input(mode_axis)
            if mode_cmd < -0.5:
                state_machine.set_flight_mode('mission')
            else:
                state_machine.set_flight_mode('hover')

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

            # Update plots
            disp_handle.update(image_color, yaw_cmd)

            if plot_2Dpos:
                pos_handle.update(drone_state.kinematics_estimated.position.x_val, drone_state.kinematics_estimated.position.y_val)

            # Execute Controller
            controller.set_current_yaw(get_yaw_from_orientation(drone_state.kinematics_estimated.orientation))
            controller.step(yaw_cmd, state_machine.get_flight_mode())

            # for CV plotting
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                disp_handle.is_active = False
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