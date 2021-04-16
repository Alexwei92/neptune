import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import threading
import cv2
import time

class Joystick():
    '''
    Read command from RC/Joystick
    '''
    def __init__(self, joystick_id=0):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(joystick_id)
        print('Load {:s} successfully.'.format(self.joystick.get_name()))
        self.numaxes = self.joystick.get_numaxes()
        self.joystick.init()

    def get_input(self, axis):
        pygame.event.pump()
        if axis > self.numaxes-1:
            self.clean()
            raise Exception('Exceed the number of axes on the joystick.')
        else:
            value = self.joystick.get_axis(axis)
            # for an imperfect neutral position 
            if abs(value) < 0.01:
                value = 0.0
            return value

    def clean(self):
        pygame.joystick.quit()
        pygame.quit()

class Joystick_fake():
    """
    A fake joystick to control the agent from keyboard
    """
    def __init__(self, joystick_id=0):
        print('Load fake joystick successfully!')
        # Multi-threading process to read keyboard input
        self.yaw_axis = 3
        self.mode_axis= 5
        self.type_axis= 6  
        self.yaw_out = 0
        self.mode_out = 0
        self.type_out = 0    
        self.is_active = True

    def get_input(self, axis):
        if axis == self.yaw_axis:
            return self.yaw_out
        elif axis == self.mode_axis:
            return self.mode_out
        elif axis == self.type_axis:
            return self.type_out

    def run(self):
        loop_rate = 10
        while self.is_active:
            tic = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF             
            if (key == 81):
                self.yaw_out = -0.5

            if (key == 84):
                self.yaw_out = 0.0 

            if (key == 83):
                self.yaw_out = 0.5

            if (key == ord('w')):
                self.mode_out = -1

            if (key == ord('s')):
                self.mode_out = 1

            if (key == ord('a')):
                self.type_out = -1

            if (key == ord('d')):
                self.type_out = 0
   
            # Ensure that the loop is running at a fixed rate
            elapsed_time = time.perf_counter() - tic
            if (1./loop_rate - elapsed_time) < 0.0:
                pass
            else:
                time.sleep(1./loop_rate - elapsed_time)

    def clean(self):
        pass