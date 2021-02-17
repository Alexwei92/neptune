import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

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
            return value # round to 8 decimal digits

    def clean(self):
        pygame.joystick.quit()
        pygame.quit()