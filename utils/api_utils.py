import setup_path
import cv2
import time, datetime
import os
import csv
import numpy as np

import airsim
from utils import plot_with_cmd, plot_with_heading, plot_without_heading

# Generate random pose
def generate_random_pose(initial_pose):
    position = airsim.Vector3r(1*np.random.rand()+initial_pose[0], 1*np.random.rand()+initial_pose[1], -initial_pose[2])
    orientation = airsim.to_quaternion(0, 0, np.pi*np.random.rand()+initial_pose[3])
    intial_pose = airsim.Pose(position, orientation)
    return intial_pose

# Get yaw from orientation
def get_yaw_from_orientation(orientation):
    pitch, roll, yaw = airsim.to_eularian_angles(orientation) # in radians
    return yaw 

# Get the color and depth images in numpy.array
def get_camera_images(client, image_size):
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
    return image_color, image_depth

class StateMachine():
    '''
    A high-level state machine
    '''
    # Init
    def __init__(self, agent_type='none', dagger_type='none', train_mode='train'):
        self.flight_mode = 'idle' # {idle, hover, mission}
        self.agent_type = agent_type 
        self.dagger_type = dagger_type
        self.train_mode = train_mode
        self.is_expert = True
        self.has_collided = False

    # Collision
    def check_collision(self, has_collided):
        if has_collided:
            print('Collision occurred! Reset to random initial state.')
            self.has_collided = True
            self.flight_mode = 'hover' # Hover by default
            print('Please reset the stick to its idle position first!')

    # Flight Mode
    def set_flight_mode(self, x):
        if x < -0.5:
            new_mode = 'mission'
        else:
            new_mode = 'hover'

        if self.has_collided and new_mode=='mission':
            # force to start from hover when reset
            pass
        else:
            self.flight_mode = new_mode
            self.has_collided = False

    def get_flight_mode(self):
        return self.flight_mode

    # Controller Type
    def set_controller_type(self, x):
        if x < -0.5 and self.agent_type != 'none':
            self.is_expert = False
        else:
            self.is_expert = True

    def get_controller_type(self):
        return self.is_expert

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

    def save_csv(self, timestamp, state, cmd):
        values = [
            str(timestamp), # timestamp
            state.position.x_val, # pos_x
            state.position.y_val, # pos_y
            state.position.z_val, # pos_z
            get_yaw_from_orientation(state.orientation), # yaw
            state.angular_velocity.z_val, # yaw_rate
            cmd, # yaw_cmd
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
    def __init__(self, image_size, max_yawRate, loop_rate=15, plot_heading=False, plot_cmd=False):
        self.is_active = True
        self.image = np.zeros((image_size[0], image_size[1], 4))
        self.max_yawRate = max_yawRate
        self.loop_rate = loop_rate
        self.plot_heading = plot_heading
        self.plot_cmd = plot_cmd

        self.heading = 0.0
        self.t_old = time.time()
        self.cmd = 0.0
        self.win_name = 'disp'
        self.is_expert = True
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def update_heading(self, cmd):
        t_new = time.time()
        self.heading = cmd * self.max_yawRate * (t_new - self.t_old)
        self.t_old = t_new

    def update_cmd(self, cmd):
        self.cmd = cmd

    def update_image(self, image):
        self.image = image

    def update(self, image, cmd, is_expert):
        self.update_image(image)
        if self.plot_heading:
            self.update_heading(cmd)
        if self.plot_cmd:
            self.update_cmd(cmd)
        self.is_expert = is_expert

    def run(self):
        t_old = time.time()
        while self.is_active:
            start_time = time.time()
            if self.plot_heading:
                plot_with_heading(self.win_name, self.image, self.heading, self.is_expert) 
            elif self.plot_cmd:
                plot_with_cmd(self.win_name, self.image, self.cmd/2.0, self.is_expert)
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


class Controller():
    '''
    A high level controller class
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.client = kwargs.get('client')
        self.forward_speed = kwargs.get('forward_speed')
        self.height = kwargs.get('height')
        self.max_yawRate = kwargs.get('max_yawRate')
        self.current_yaw = 0.0 # in radian

    def set_current_yaw(self, yaw):
        self.current_yaw = yaw

    def step(self, yaw_cmd, flight_mode):
        if flight_mode == 'hover':
            # hover
            self.client.rotateByYawRateAsync(yaw_rate=yaw_cmd * self.max_yawRate,
                        duration=1)

        elif flight_mode == 'mission':
            # forward flight
            vx = self.forward_speed * np.cos(self.current_yaw)
            vy = self.forward_speed * np.sin(self.current_yaw)
            self.client.moveByVelocityZAsync(vx=vx, vy=vy, 
                                            z=-self.height,
                                            duration=1,
                                            yaw_mode=airsim.YawMode(True, yaw_cmd * self.max_yawRate))

        else:
            print('Unknown flight_mode: ', flight_mode)
            raise Exception
