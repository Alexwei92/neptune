from controller import BaseCtrl

class ExpertCtrl(BaseCtrl):
    '''
    Expert Controller
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, yaw_cmd, flight_mode):
        if flight_mode == 'hover':
            self.send_command(yaw_cmd, is_hover=True)
        elif flight_mode == 'mission':
            self.send_command(yaw_cmd, is_hover=False)
        else:
            print('Unknown flight_mode: ', flight_mode)
            raise Exception