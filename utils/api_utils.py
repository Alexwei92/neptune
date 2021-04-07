import cv2
import time, datetime
import os
import shutil
import csv
import numpy as np
import random
import colorama

import airsim
from utils import *
from controller import *

PRECISION = 8 # decimal digits
colorama.init()

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

# Get the color and depth images in numpy.array format
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
        # print('\033[37;42m' + msg + '\033[m')
        print(colorama.Back.GREEN + colorama.Fore.WHITE + msg + colorama.Style.RESET_ALL)
    elif type == 2:
        # print('\033[33m' + '[WARNING] ' + msg + '\033[m')
        print(colorama.Fore.YELLOW + '[WARNING] ' + msg + colorama.Style.RESET_ALL)
    elif type == 3:
        # print('\033[31m' + '[ERROR] ' + msg + '\033[m')
        print(colorama.Fore.RED + '[ERROR] ' + msg + colorama.Style.RESET_ALL)
    else:
        print(msg)

class FastLoop():
    '''
    API Fast Loop
    '''
    # Init
    def __init__(self, **kwargs):
        # simulation
        self.image_size = kwargs.get('image_size')
        self.initial_pose = kwargs.get('initial_pose')
        self.loop_rate = kwargs.get('loop_rate')
        self.train_mode = kwargs.get('train_mode')
        
        # controller
        self.agent_type = kwargs.get('agent_type') 
        self.dagger_type = kwargs.get('dagger_type') 
        self.forward_speed = kwargs.get('forward_speed')
        self.max_yawRate = kwargs.get('max_yawRate')
        self.mission_height = kwargs.get('mission_height') 

        # joystick
        self.joy = kwargs.get('joystick')
        self.yaw_axis = kwargs.get('yaw_axis')
        self.mode_axis = kwargs.get('mode_axis')
        self.type_axis = kwargs.get('type_axis')

        # rangefinder
        self.use_rangefinder = kwargs.get('use_rangefinder')
        if self.use_rangefinder:
            self.rangefinder = Rangefinder(cutoff_freq=2.0, sample_freq=self.loop_rate)
        
        # states and camera images
        self.drone_state = None
        self.estimated_height = 0.0
        self.image_color = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        self.image_depth = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        
        # status and cmd
        self.flight_mode = 'hover' # {hover, mission}
        self.has_collided = False
        self.is_active = True
        self.is_expert = True
        self.pilot_cmd = 0.0
        self.agent_cmd = 0.0
        self.trigger_reset = False # to trigger external reset function
        self.force_reset = False # from external to force reset
        self.manual_stop = False # if pilot manually stop the mission
        self.virtual_crash = False # for a virtual crash in HG-Dagger

        # Connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Enable API control
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Take-off
        self.client.takeoffAsync()

        # Initial pose
        self.client.simSetVehiclePose(add_offset_to_pose(random.choice(self.initial_pose)), ignore_collison=True)
        time.sleep(0.5)

        # Controller Init
        self.controller = Controller(self.client, 
                    self.forward_speed,
                    self.mission_height,
                    self.max_yawRate,
                    self.use_rangefinder)


    # Collision
    def set_collision(self):
        self.trigger_reset = True # trigger external reset function
        if self.flight_mode is 'mission':
            print_msg('Collision occurred! API is reset to a random initial pose.', type=3)
            print_msg('Please reset the stick to its neutral position first!', type=2)
            self.has_collided = True
            self.flight_mode = 'hover' # Hover by default
            print_msg('{:s} flight mode'.format(self.flight_mode.capitalize()))
        
    # Flight Mode
    def set_flight_mode(self, joy_input):
        if joy_input < -0.5:
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
            if new_mode is not self.flight_mode:
                self.flight_mode = new_mode
                print_msg('{:s} flight mode'.format(new_mode.capitalize()))
                if new_mode is 'hover':
                    self.manual_stop = True # which pilot stop the mission

    # Controller Type
    def set_controller_type(self, joy_input):
        if self.agent_type != 'none':
            if joy_input < -0.5:
                is_expert = False
            else:
                is_expert = True
            
            if is_expert != self.is_expert:
                self.is_expert = is_expert
                if is_expert:
                    print_msg('Switch to manual control')
                    # if self.flight_mode is 'mission' and self.dagger_type is 'hg':
                    #     self.virtual_crash = True # when the pilot think it may crash (HG-dagger)
                else:
                    print_msg('Switch to agent control')
            
    def get_yaw_rate(self):
        return self.drone_state.kinematics_estimated.angular_velocity.z_val

    # Fast loop
    def run(self):
        while self.is_active:
            start_time = time.perf_counter()

            # Check collision
            if self.client.simGetCollisionInfo().has_collided:
                self.reset_api()

            # Force Reset
            if self.force_reset:
                self.reset_api()
                self.force_reset = False

            # Update pilot yaw command from RC/joystick
            self.pilot_cmd = round(self.joy.get_input(self.yaw_axis), PRECISION) # round to precision
        
            # Update flight mode from RC/joystick
            self.set_flight_mode(self.joy.get_input(self.mode_axis))
            
            # Update controller type from RC/joystick
            self.set_controller_type(self.joy.get_input(self.type_axis))

            # Update state
            try:
                self.drone_state = self.client.getMultirotorState()
            except:
                pass # buffer issue
            
            # Update yaw
            current_yaw = get_yaw_from_orientation(self.drone_state.kinematics_estimated.orientation)
            self.controller.set_current_yaw(current_yaw)

            # Update rangefinder
            if self.use_rangefinder:
                try:
                    self.rangefinder_height = self.rangefinder.update(self.client.getDistanceSensorData().distance)
                except:
                    pass # buffer issue
                estimated_height = self.rangefinder_height
                # Update z velocity
                self.controller.set_current_zvelocity(self.drone_state.kinematics_estimated.linear_velocity.z_val)
            else:
                estimated_height = self.mission_height

            # Update images
            camera_response = self.client.simGetImages([
                # uncompressed RGB array bytes 
                airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=False),
                # floating point uncompressed image
                airsim.ImageRequest('1', airsim.ImageType.DepthVis, pixels_as_float=True, compress=False)])
            try:
                self.image_color, self.image_depth = get_camera_images(camera_response, self.image_size)
            except:
                pass # buffer issue

            # Update controller
            if self.is_expert or self.flight_mode == 'hover':
                self.agent_cmd = self.pilot_cmd
                try:
                    self.controller.step(self.pilot_cmd, estimated_height, self.flight_mode)
                except:
                    pass
            else:
                self.controller.step(self.agent_cmd, estimated_height, 'mission')

            # Force the loop rate
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 1./self.loop_rate:
                print_msg('The fast loop is running at {:.2f} Hz, expected {:.2f} Hz!'.format(1./elapsed_time, self.loop_rate), type=2)
            else:
                time.sleep(1./self.loop_rate - elapsed_time) 
        
        self.clean()

    # Reset the API after collision
    def reset_api(self):
        self.set_collision()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simSetVehiclePose(add_offset_to_pose(random.choice(self.initial_pose)), ignore_collison=True)      

        if self.use_rangefinder:
            self.rangefinder.reset()
            self.controller.reset_pid()

    def clean(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()

class Logger():
    '''
    Data logger
    '''
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        self.root_dir = root_dir
        # Configure folder
        self.configure_folder()

    def configure_folder(self):
        folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")
        folder_path = self.root_dir + '/' + folder_name
        os.makedirs(folder_path + '/' + 'color')
        os.makedirs(folder_path + '/' + 'depth')
        self.folder_path = folder_path

        self.file = open(folder_path + '/' + 'airsim.csv', 'w', newline='')
        self.filewriter = csv.writer(self.file, delimiter=',')
        self.filewriter.writerow(['timestamp','pos_x','pos_y','pos_z','yaw','yaw_rate','yaw_cmd','flag'])
        self.flag = 0
        self.index = 0

    def update_flag(self, is_expert):
        if is_expert:
            self.flag = 0
        else:
            self.flag = 1

    def save_image(self, folder, image):
        cv2.imwrite(os.path.join(self.folder_path + '/' + folder, "%07i.png" % self.index), image)

    def save_csv(self, timestamp, state, cmd):
        values = [
            str(timestamp), # timestamp, ns
            round(state.position.x_val, PRECISION), # pos_x, m
            round(state.position.y_val, PRECISION), # pos_y, m
            round(state.position.z_val, PRECISION), # pos_z, m
            round(get_yaw_from_orientation(state.orientation), PRECISION), # yaw, rad
            round(state.angular_velocity.z_val, PRECISION), # yaw_rate, rad/s
            round(cmd, PRECISION), # yaw_cmd [-1,1]
            self.flag] # flag
        self.filewriter.writerow(values)
        self.index += 1

    def reset_folder(self, status='crashed'):
        self.filewriter.writerow([status])
        self.clean()
        self.configure_folder()
        
    def clean(self):
        self.file.close()
        if self.index == 0:
            # delete the folder if no content
            shutil.rmtree(self.folder_path)

class Display():
    '''
    For displaying the window(s)
    '''
    def __init__(self, loop_rate, image_size, max_yawRate, plot_heading=False, plot_cmd=False, plot_trajectory=False):
        self.image = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)
        self.max_yawRate = max_yawRate
        self.loop_rate = loop_rate
        self.plot_heading = plot_heading
        self.plot_cmd = plot_cmd
        # cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
        if plot_trajectory:
            fig, ax = plt.subplots()
            self.trajectory_handle = DynamicPlot(fig, ax, max_width=120*loop_rate) # plot 2 minutes trajectory
        
        self.is_active = True
        self.is_expert = True
        self.heading = 0.0
        self.cmd = 0.0
        self.t_old = time.perf_counter()      

    def update(self, image, cmd, is_expert):
        self.image = image
        self.is_expert = is_expert
        if self.plot_heading:
            t_new = time.perf_counter()
            self.heading = cmd * self.max_yawRate * (t_new - self.t_old)
            self.t_old = t_new
        if self.plot_cmd:
            self.cmd = cmd

    def update_trajctory(self, x, y):
        self.trajectory_handle.update(round(x, 3), round(y, 3))

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
            if elapsed_time > 1./self.loop_rate:
                # this should never happen
                print_msg('The visualize loop rate is too high, consider to reduce the rate!', type=2)
            else:
                time.sleep(1./self.loop_rate - elapsed_time) 
        
        self.clean()
    
    def clean(self):
        # cv2.destroyAllWindows()
        pass

class Controller():
    '''
    A high-level controller class
    '''
    def __init__(self, client, forward_speed, height, max_yawRate, use_rangefinder=False):
        self.client = client
        self.forward_speed = forward_speed
        self.height = height
        self.max_yawRate = max_yawRate
        self.current_yaw = 0.0 # in radian
        self.use_rangefinder = use_rangefinder # use rangefinder
        if use_rangefinder:
            self.z_velocity = 0.0
            self.zpos_pid = PIDController(kp=0.25, ki=0.0, kd=0.0, scale=6.0)
            self.zvel_pid = PIDController(kp=2.0, ki=2.0, kd=0.0, scale=1.0)

        self.cmd_history = np.zeros((20,)) # record history cmd in memory 

    def set_current_yaw(self, yaw):
        self.current_yaw = yaw

    def set_current_zvelocity(self, z_velocity):
        self.z_velocity = -z_velocity

    def step(self, cmd, estimated_height, flight_mode):
        self.cmd_history = np.append(np.delete(self.cmd_history, 0), cmd) # [i-N,...,i-1]
        yawRate = cmd * self.max_yawRate

        if self.use_rangefinder:
            zpos_output = self.zpos_pid.update(self.height, estimated_height)
            zvel_output = -self.zvel_pid.update(zpos_output, self.z_velocity) # z axis is reversed
            throttle = 0.5 + zvel_output

        if flight_mode == 'hover':
            # hover
            if self.use_rangefinder:
                self.client.moveByVelocityAsync(vx=0, vy=0, 
                                            vz=zvel_output,
                                            duration=1,
                                            yaw_mode=airsim.YawMode(True, yawRate))
            else:
                self.client.rotateByYawRateAsync(yaw_rate=yawRate, duration=1)
        
        elif flight_mode == 'mission':
            # forward flight
            vx = self.forward_speed * np.cos(self.current_yaw)
            vy = self.forward_speed * np.sin(self.current_yaw)
            if self.use_rangefinder:
                self.client.moveByVelocityAsync(vx=vx, vy=vy, 
                                            vz=zvel_output,
                                            duration=1,
                                            yaw_mode=airsim.YawMode(True, yawRate))
            else:
                self.client.moveByVelocityZAsync(vx=vx, vy=vy, 
                                                z=-self.height,
                                                duration=1,
                                                yaw_mode=airsim.YawMode(True, yawRate))
        else:
            raise Exception('Unknown flight_mode: ' + flight_mode)

    def reset_pid(self):
        self.zpos_pid.reset()
        self.zvel_pid.reset()
        self.z_velocity = 0.0

    def reset_cmd_history(self):
        self.cmd_history.fill(0.0)

class Rangefinder():
    '''
    Rangefinder (distance sensor) class
    '''
    def __init__(self, cutoff_freq=2, sample_freq=10):
        self.lowpassfilter = SecondOrderLowPass(cutoff_freq, sample_freq)

    def update(self, value):
        return self.lowpassfilter.update(value)
        
    def reset(self):
        self.lowpassfilter.reset()