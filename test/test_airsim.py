import setup_path
import cv2
import time, datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import airsim
from utils import *

class StateMachine():
    def __init__(self):
        self.position = airsim.Vector3r() # {x, y, z}
        self.orientation = airsim.Vector3r() # {roll, pitch, yaw}
        self.mode = 0 # {0: idle, 1: hover, 2: mission}
        self.has_collided = False

    def set_collision(self, has_collided):
        if has_collided:
            print('Collision occurred! Reset to initial state')
            self.has_collided = True
            self.mode = 1

    def set_state(self, kinematics_estimated):
        self.position = kinematics_estimated.position
        pitch, roll, yaw = airsim.to_eularian_angles(kinematics_estimated.orientation)
        self.orientation = airsim.Vector3r(roll, pitch, yaw)

    def set_mode(self, mode):
        if self.has_collided and mode==2:
            pass
        else:
            self.mode = mode
            self.has_collided = False

    def get_mode(self):
        return self.mode

    def get_yawRad(self):
        return self.orientation.z_val

class Controller():
    def __init__(self, client, forward_speed, height, max_yawRate):
        self.client = client
        self.forward_speed = forward_speed
        self.height = height
        self.max_yawRate = max_yawRate
        self.current_yaw = 0.0 # in radian
        self.duration = 5

    def set_current_yaw(self, yaw):
        self.current_yaw = yaw

    def step(self, yaw_cmd, mode=4):
        if mode == 0: # idle
            pass
        if mode == 1: # hover
            self.client.rotateByYawRateAsync(yaw_rate=yaw_cmd * self.max_yawRate,
                                             duration=self.duration)
        if mode == 2: # mission
            vx = forward_speed * np.cos(self.current_yaw)
            vy = forward_speed * np.sin(self.current_yaw)
            self.client.moveByVelocityZAsync(vx=vx, vy=vy, 
                                            z=-self.height,
                                            duration=self.duration,
                                            yaw_mode=airsim.YawMode(True, yaw_cmd * self.max_yawRate))

class Logger():
    '''
    Data Logger
    '''
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        folder_name = datetime.datetime.now().strftime("%Y_%h_%d_%H_%M_%S")
        path = root_dir+'/'+folder_name
        os.makedirs(path+'/'+'color')
        os.makedirs(path+'/'+'depth')
        self.path = path

        self.f = open(path+'/'+'airsim', 'w', newline='')
        self.filewriter = csv.writer(self.f, delimiter = ',')
        self.filewriter.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw_cmd', 'trial'])
        self.trial = 1
        self.index = 0

    def save_image(self, type, img):
        cv2.imwrite(os.path.join(self.path+'/'+type, "frame%07i.jpg" % self.index), img)

    def save_csv(self, timestamp, position, yaw_cmd):
        values = [str(timestamp), position.x_val, position.y_val, position.z_val, yaw_cmd, self.trial]
        self.filewriter.writerow(values)
        self.index += 1

    def clean(self):
        self.f.close()


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

# Parameters
loop_rate       = 10.0   # Hz
max_yawRate     = 45.0  # deg/s
forward_speed   = 1.5   # m/s
height          = 5.0   # m

# For visualization
fig = plt.figure()
ax = plt.subplot()
pos_handle = DynamicPlot(fig, ax, max_width=120*loop_rate)
# cv2.namedWindow("mirror", cv2.WINDOW_NORMAL)

# Joystick/RC instance
joy        = Joystick(0) 
yaw_axis   = 3
speed_axis = 5
mode_axis  = 7

# State Machine
state_machine = StateMachine()

# Controller
controller = Controller(client,
                        forward_speed=forward_speed, 
                        height=height, 
                        max_yawRate=max_yawRate)

# Data logger
data_logger = Logger('D:/airsim_data')

'''
Main loop
'''
try:
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Taking off, please wait...")
    client.takeoffAsync().join()
    state_machine.set_mode(1) # hover
    print("Start the loop")

    while True:
        now = time.time() # loop start time

        # Check collision
        state_machine.set_collision(client.simGetCollisionInfo().has_collided)
        if client.simGetCollisionInfo().has_collided:
            client.reset()
            client.enableApiControl(True)
            client.simSetVehiclePose(generate_random_pose(height), True)
            data_logger.trial += 1

        # Get states
        state = client.getMultirotorState()
        state_machine.set_state(state.kinematics_estimated)
        
        # Update yaw command
        yaw_cmd = joy.get_input(yaw_axis)

        # Update mode
        mode_cmd = joy.get_input(mode_axis)
        if mode_cmd < -0.5:
            state_machine.set_mode(2)
        else:
            state_machine.set_mode(1)

        # Get the rgb image in FPV
        camera_color = client.simGetImage("0", airsim.ImageType.Scene)
        camera_depth = client.simGetImage("0", airsim.ImageType.DepthVis)
        pngImg_color = cv2.imdecode(np.frombuffer(camera_color, np.int8), cv2.IMREAD_UNCHANGED)
        pngImg_depth = cv2.imdecode(np.frombuffer(camera_depth, np.int8), cv2.IMREAD_UNCHANGED)
        # grayImg_color = cv2.cvtColor(pngImg_color, cv2.COLOR_BGR2GRAY)

        # Data logging
        if state_machine.get_mode() == 2:
            data_logger.save_image('color', pngImg_color)
            data_logger.save_image('depth', pngImg_depth)
            data_logger.save_csv(state.timestamp, state.kinematics_estimated.position, yaw_cmd)

        # Update plots
        pos_handle.update(state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val)
        # plot_with_heading(pngImg_color, yaw_cmd/2.0)

        # Send command to the vehicle
        controller.set_current_yaw(state_machine.get_yawRad())
        controller.step(yaw_cmd, mode=state_machine.get_mode())

        # for CV plotting
        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            client.reset()
            break

        # Ensure that the loop is running at a fixed rate
        elapsed_time = time.time() - now
        if (1./loop_rate - elapsed_time) < 0.0:
            print('Warning: the loop is too fast, please reduce the rate!')
        else:
            time.sleep(1./loop_rate - elapsed_time)

except:
    data_logger.clean()
    # client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)
    joy.clean()