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
from controller import ExpertCtrl

class StateMachine():
    '''
    A high-level state machine
    '''
    # Init
    def __init__(self, controller_type='expert', current_mode='training'):
        self.position = airsim.Vector3r() # {x, y, z}
        self.orientation = airsim.Vector3r() # {roll, pitch, yaw}
        self.flight_mode = 'idle' # {0: idle, 1: hover, 2: mission}
        self.controller_type = controller_type # {0: expert, 1: agent, 2:hybrid}
        self.current_mode = current_mode # {0: training, 1: test}
        self.has_collided = False

    # Collision
    def check_collision(self, has_collided):
        if has_collided:
            print('Collision occurred! Reset to random initial state.')
            self.has_collided = True
            self.flight_mode = 'hover' # Hover by default

    # Pose
    def set_pose(self, kinematics_estimated):
        self.position = kinematics_estimated.position
        pitch, roll, yaw = airsim.to_eularian_angles(kinematics_estimated.orientation)
        self.orientation = airsim.Vector3r(roll, pitch, yaw)

    def get_pose(self):
        return self.position, self.orientation

    def get_position(self):
        return self.position

    def get_yawRad(self):
        return self.orientation.z_val

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
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")
        path = root_dir+'/'+folder_name
        os.makedirs(path+'/'+'color')
        os.makedirs(path+'/'+'depth')
        self.path = path

        self.f = open(path+'/'+'airsim.csv', 'w', newline='')
        self.filewriter = csv.writer(self.f, delimiter = ',')
        self.filewriter.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'yaw_cmd', 'trial'])
        self.trial = 1
        self.index = 0

    def save_image(self, type, img):
        cv2.imwrite(os.path.join(self.path+'/'+type, "%07i.png" % self.index), img)

    def save_csv(self, timestamp, position, yaw, yaw_cmd):
        values = [str(timestamp), position.x_val, position.y_val, position.z_val, yaw, yaw_cmd, self.trial]
        self.filewriter.writerow(values)
        self.index += 1

    def clean(self):
        self.f.close()


class Display():
    '''
    For displaying on the window
    '''
    def __init__(self, image_size, loop_rate=15):
        self.is_active = True
        self.image = np.zeros((image_size[0], image_size[1], 4))
        self.bar = 0.0
        self.loop_rate = loop_rate

    def update_bar(self, bar):
        self.bar = bar

    def update_image(self, image):
        self.image = image

    def run(self):
        while self.is_active:
            start_time = time.time()
            plot_with_heading(self.image, self.bar/2.0)
            elapsed_time = time.time() - start_time

            if (1./self.loop_rate - elapsed_time) < 0.0:
                print('[WARNING] The visualize loop rate is too high, consider to reduce the rate!')
            else:
                time.sleep(1./self.loop_rate - elapsed_time)
        self.clean()
    
    def clean(self):
        cv2.destroyAllWindows()

# Generate random pose
def generate_random_pose(height):
    position = airsim.Vector3r(2.5*np.random.rand(), 2.5*np.random.rand(), -height)
    orientation = airsim.to_quaternion(0, 0, np.pi*np.random.rand())
    intial_pose = airsim.Pose(position, orientation)
    return intial_pose

'''
Initialize
'''
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Read YAML configurations
with open('config.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(e)

# Simulation settings
loop_rate = config['sim_params']['loop_rate']
max_yawRate = config['sim_params']['max_yawRate']
forward_speed = config['sim_params']['forward_speed']
height = config['sim_params']['height']
image_size = eval(config['sim_params']['image_size'])

# Joystick/RC settings
joy = Joystick(config['rc_params']['device_id']) 
yaw_axis = config['rc_params']['yaw_axis']
speed_axis = config['rc_params']['speed_axis']
mode_axis = config['rc_params']['mode_axis']

# Visualize settings
disp_handle = Display(image_size=image_size, loop_rate=loop_rate)
if config['visualize_params']['plot_2Dpos']:
    fig, ax = plt.subplots()
    pos_handle = DynamicPlot(fig, ax, max_width=120*loop_rate)

if config['visualize_params']['plot_heading']:
    cv2.namedWindow("mirror", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("mirror", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("mirror",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


# State Machine
state_machine = StateMachine(
    controller_type=config['sim_params']['controller_type'],
    current_mode='training',
)

# Controller
controller = ExpertCtrl(client=client,
                        forward_speed=forward_speed, 
                        height=height, 
                        max_yawRate=max_yawRate)

# # Data logger
# data_logger = Logger(os.path.join(setup_path.parent_dir, 'my_datasets/nh'))

'''
Main loop
'''
try:
    client.enableApiControl(True)
    client.armDisarm(True)

    init_position = airsim.Vector3r(115, 181, -height)
    init_orientation = airsim.to_quaternion(0, 0, np.pi*np.random.rand())
    intial_pose = airsim.Pose(init_position, init_orientation)
    client.simSetVehiclePose(intial_pose, True)
    
    print("Taking off, please wait...")
    client.takeoffAsync().join()
    state_machine.set_flight_mode('hover')
    controller.is_active = True
    print("Start the loop")

    # multithreading
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
            # data_logger.trial += 1

        # Get Multirotor states
        state = client.getMultirotorState()
        state_machine.set_pose(state.kinematics_estimated)
        
        # Update yaw command from RC/joystick
        yaw_cmd = joy.get_input(yaw_axis)

        # Update flight mode from RC/joystick
        mode_cmd = joy.get_input(mode_axis)
        if mode_cmd < -0.5:
            state_machine.set_flight_mode('mission')
        else:
            state_machine.set_flight_mode('hover')

        # Get the rgb image in FPV
        camera_color = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        camera_color = np.fromstring(camera_color.image_data_uint8, dtype=np.uint8)
        camera_color = camera_color.reshape(image_size[0], image_size[1], 3)

        camera_depth = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.DepthVis, False, False)])[0]
        # print(camera_depth.image_type)
        camera_depth = np.fromstring(camera_depth.image_data_uint8, dtype=np.uint8)
        print(camera_depth.shape)
        camera_color = camera_color.reshape(image_size[0], image_size[1], 3)

        # camera_depth = client.simGetImage("0", airsim.ImageType.DepthVis)
        # pngImg_depth = cv2.imdecode(np.frombuffer(camera_depth, np.int8), cv2.IMREAD_UNCHANGED)
        # print(pngImg_depth.shape)
        # grayImg_color = cv2.cvtColor(pngImg_color, cv2.COLOR_BGR2GRAY)


        # # Data logging
        # if state_machine.get_flight_mode() == 'mission':
        #     data_logger.save_image('color', pngImg_color)
        #     data_logger.save_image('depth', pngImg_depth)
        #     data_logger.save_csv(state.timestamp, state.kinematics_estimated.position, state_machine.get_yawRad(), yaw_cmd)

        # Update plots
        # pos_handle.update(state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val)
        disp_handle.update_bar(yaw_cmd)
        disp_handle.update_image(camera_color)

        # Send command to the vehicle
        controller.set_current_yaw(state_machine.get_yawRad())
        controller.step(yaw_cmd, state_machine.get_flight_mode())

        # for CV plotting
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            disp_handle.is_active = False
            break

        # Ensure that the loop is running at a fixed rate
        elapsed_time = time.time() - now
        print(1./elapsed_time)
        if (1./loop_rate - elapsed_time) < 0.0:
            print('[WARNING] The main loop rate is too high, consider to reduce the rate!')
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
    # data_logger.clean()
    # client.armDisarm(False)
    client.enableApiControl(False)
    client.reset()
    print('Exit the program successfully!')