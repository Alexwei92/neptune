import cv2
import time, datetime
import os
import shutil
import csv
import numpy as np

import airsim
from utils import plot_with_cmd, plot_with_heading, plot_without_heading

# Add random offset to pose
def add_offset_to_pose(pose, pos_offset=1, yaw_offset=np.pi*2):
    position = airsim.Vector3r(pos_offset*np.random.rand()+pose[0], pos_offset*np.random.rand()+pose[1], -pose[2])
    orientation = airsim.to_quaternion(0, 0, yaw_offset*np.random.rand()+pose[3])
    pose_with_offset = airsim.Pose(position, orientation)
    return pose_with_offset

# Get yaw from orientation
def get_yaw_from_orientation(orientation):
    pitch, roll, yaw = airsim.to_eularian_angles(orientation) # in radians
    return yaw 

# Get the color and depth images in numpy.array
def get_camera_images(camera_response, image_size):
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

# Colored print out
def print_msg(msg, type=0):
    # type = {0:Default, 1:Status, 2:Warning, 3:Error}
    if type == 1:
        print('\033[37;42m' + msg + '\033[m')
    elif type == 2:
        print('\033[33m' + '[WARNING] ' + msg + '\033[m')
    elif type == 3:
        print('\033[31m' + '[ERROR] ' + msg + '\033[m')
    else:
        print(msg)

# Reset the environment after collision
def reset_environment(client, state_machine, initial_pose, controller_agent):
    state_machine.set_collision()
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simSetVehiclePose(add_offset_to_pose(initial_pose), ignore_collison=True)

    if state_machine.agent_type == 'reg':
        controller_agent.reset_prvs()
####
class StateMachine():
    '''
    A high-level state machine
    '''
    # Init
    def __init__(self, agent_type='none', train_mode='train'):
        self.flight_mode = 'idle' # {idle, hover, mission}
        self.agent_type = agent_type 
        self.train_mode = train_mode
        self.is_expert = True
        self.has_collided = False

    # Collision
    def set_collision(self):
        print('Collision occurred! API is reset to random initial pose.')
        self.has_collided = True
        self.flight_mode = 'hover' # Hover by default
        print_msg('In {:s} flight mode'.format(self.flight_mode))
        print_msg('Please reset the stick to its idle position first!', type=2)

    # Flight Mode
    def set_flight_mode(self, x):
        if x < -0.5:
            new_mode = 'mission'
        else:
            new_mode = 'hover'
    
        if self.has_collided:
            if new_mode is 'mission':
                pass # force to start from hover when reset
            else:
                self.has_collided = False
                print_msg("Ready to fly!", type=1)
        else:
            if new_mode != self.flight_mode:
                self.flight_mode = new_mode
                print_msg('In {:s} flight mode'.format(new_mode))

    def get_flight_mode(self):
        return self.flight_mode

    # Controller Type
    def set_controller_type(self, x):
        if self.agent_type != 'none':
            if x < -0.5:
                is_expert = False
            else:
                is_expert = True
            
            if is_expert != self.is_expert:
                self.is_expert = is_expert
                if is_expert:
                    print_msg('Switch to manual control')
                else:
                    print_msg('Switch to agent control')
            

    def get_controller_type(self):
        return self.is_expert

class Logger():
    '''
    Data logger
    '''
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")
        folder_path = root_dir+'/'+folder_name
        os.makedirs(folder_path+'/'+'color')
        os.makedirs(folder_path+'/'+'depth')
        self.folder_path = folder_path

        self.file = open(folder_path+'/'+'airsim.csv', 'w', newline='')
        self.filewriter = csv.writer(self.file, delimiter = ',')
        self.filewriter.writerow(['timestamp','pos_x','pos_y','pos_z','yaw','yaw_rate','yaw_cmd','crash_count','image_name'])
        self.crash_count = 0
        self.index = 0

    def save_image(self, folder, img):
        cv2.imwrite(os.path.join(self.folder_path+'/'+folder, "%07i.png" % self.index), img)

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
        self.file.close()
        if self.index == 0:
            shutil.rmtree(self.folder_path)

class Display():
    '''
    For displaying the window(s)
    '''
    def __init__(self, image_size, max_yawRate, loop_rate=15, plot_heading=False, plot_cmd=False):
        self.is_active = True
        self.is_expert = True
        self.image = np.zeros((image_size[0], image_size[1], 4))
        self.max_yawRate = max_yawRate
        self.dt = 1./loop_rate
        self.plot_heading = plot_heading
        self.plot_cmd = plot_cmd
        cv2.namedWindow('disp', cv2.WINDOW_NORMAL)

        self.heading = 0.0
        self.cmd = 0.0
        self.t_old = time.perf_counter()      

    def update(self, image, cmd, is_expert):
        self.image = image
        self.is_expert = is_expert
        if self.plot_heading:
            t_new = time.perf_counter()
            self.heading = cmd * self.max_yawRate * (t_new - self.t_old)
            # print(t_new - self.t_old)
            self.t_old = t_new
        if self.plot_cmd:
            self.cmd = cmd

    def run(self):
        while self.is_active:
            start_time = time.perf_counter()
            if self.plot_heading:
                plot_with_heading('disp', self.image, self.heading, self.is_expert) 
            elif self.plot_cmd:
                plot_with_cmd('disp', self.image, self.cmd/2.0, self.is_expert)
            else:
                plot_without_heading('disp', self.image)

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > self.dt:
                # this should never happen
                print_msg('The visualize loop rate is too high, consider to reduce the rate!', type=2)
            else:
                time.sleep(self.dt - elapsed_time) 
        
        self.clean()
    
    def clean(self):
        cv2.destroyAllWindows()


class Controller():
    '''
    A high-level controller class
    '''
    def __init__(self, client, forward_speed, height, max_yawRate):
        self.client = client
        self.forward_speed = forward_speed
        self.height = height
        self.max_yawRate = max_yawRate
        self.current_yaw = 0.0 # in radian

    def set_current_yaw(self, yaw):
        self.current_yaw = yaw

    def step(self, cmd, flight_mode):
        yawRate = cmd * self.max_yawRate

        if flight_mode is 'hover':
            # hover
            self.client.rotateByYawRateAsync(yaw_rate=yawRate, duration=1)

        elif flight_mode is 'mission':
            # forward flight
            vx = self.forward_speed * np.cos(self.current_yaw)
            vy = self.forward_speed * np.sin(self.current_yaw)
            self.client.moveByVelocityZAsync(vx=vx, vy=vy, 
                                            z=-self.height,
                                            duration=1,
                                            yaw_mode=airsim.YawMode(True, yawRate))

        else:
            raise Exception('Unknown flight_mode: ' + flight_mode)
