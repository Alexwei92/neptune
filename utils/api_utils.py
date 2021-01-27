import setup_path
import cv2
import time, datetime
import os
import csv
import numpy as np

import airsim
from utils import plot_with_heading, plot_without_heading

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
            print('Please reset the stick to its idle position first!')


    # Flight Mode
    def set_flight_mode(self, flight_mode):
        if self.has_collided and flight_mode=='mission':
            # force to start from hover when reset
            pass
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