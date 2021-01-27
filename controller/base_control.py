import numpy as np
from airsim import YawMode

class BaseCtrl():
    '''
    Base Controller
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.client = kwargs.get('client')
        self.forward_speed = kwargs.get('forward_speed')
        self.height = kwargs.get('height')
        self.max_yawRate = kwargs.get('max_yawRate')
        self.is_active = False
        self.current_yaw = 0.0 # in radian

    def set_current_yaw(self, yaw):
        self.current_yaw = yaw

    def step(self, yaw_cmd, flight_mode):
        if flight_mode == 'hover':
            self.send_command(yaw_cmd, is_hover=True)
        elif flight_mode == 'mission':
            self.send_command(yaw_cmd, is_hover=False)
        else:
            print('Unknown flight_mode: ', flight_mode)
            raise Exception

    def send_command(self, yaw_cmd, is_hover=False):
        if is_hover:
            # hover
            self.client.rotateByYawRateAsync(yaw_rate=yaw_cmd * self.max_yawRate,
                                    duration=1)
        else:
            # forward flight
            vx = self.forward_speed * np.cos(self.current_yaw)
            vy = self.forward_speed * np.sin(self.current_yaw)
            self.client.moveByVelocityZAsync(vx=vx, vy=vy, 
                                            z=-self.height,
                                            duration=1,
                                            yaw_mode=YawMode(True, yaw_cmd * self.max_yawRate))
