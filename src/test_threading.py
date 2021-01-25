import pygame
import threading

class Joystick():
    '''
    Read command from RC/Joystick
    '''
    def __init__(self, joystick_id=0):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(joystick_id)
        self.joystick.init()
        self.stop = False

    def get_input(self, axis):
        pygame.event.pump()
        # print(self.joystick.get_axis(axis))
        return self.joystick.get_axis(axis)

    def clean(self):
        pygame.joystick.quit()
        pygame.quit()


    def start_loop(self, axis):
        while True:
            tmp = self.get_input(axis)
            if tmp > 0.75:
                self.stop = True

a = Joystick()
axis = 3
x = threading.Thread(target=a.start_loop, args=(axis,))
x.start()
print('Here')
print(x._stop)
print('There')
