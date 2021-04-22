"""
Evaluate the DAgger performance
"""
import csv
import cv2
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

def read_data(folder_path):
    # Read telemetry csv
    data = pandas.read_csv(os.path.join(folder_path, 'airsim.csv'))
    N = len(data) - 1 # length of data 
    status = data.iloc[-1,0]

    # Timestamp (sec)
    timestamp = data['timestamp'][:-1].to_numpy(float)
    timestamp -= timestamp[0]
    timestamp *= 1e-9

    # Yaw (rad/s)
    yaw_rate = data['yaw_rate'][:-1].to_numpy(dtype=np.float32) 

    # Yaw cmd ([-1,1])
    yaw_cmd = data['yaw_cmd'][:-1].to_numpy(dtype=np.float32) 

    # Pos X and Y (m)
    pos_x = data['pos_x'][:-1].to_numpy(dtype=np.float32)
    pos_y = data['pos_y'][:-1].to_numpy(dtype=np.float32)

    # Flag
    flag = data['flag'][:-1].to_numpy(dtype=np.float32)

    return {'pos_x':pos_x, 'pos_y':pos_y, 
            'yaw_rate':yaw_rate, 'yaw_cmd':yaw_cmd, 
            'flag':flag, 'status':status, 'timestamp':timestamp}

def calculate_distance(pos_x, pos_y):
    total_distance = 0
    last_x, last_y = pos_x[0], pos_y[0]
    for x, y in zip(pos_x, pos_y):
        total_distance += np.sqrt((x-last_x)**2 + (y-last_y)**2) 
        last_x, last_y = x, y
    return total_distance

def calculate_time(timestamp):
    return timestamp[-1] - timestamp[0]

def calculate_intervene(flag):
    return sum(flag==0) / len(flag)

def calculat_intervene_per_distance(flag, total_distance):
    return sum(flag==0) / total_distance

def process_data(data_dict):
    # total time
    time = calculate_time(data_dict['timestamp'])
    print('Total time: {:.2f} sec'.format(time))
    
    # total distance
    distance = calculate_distance(data_dict['pos_x'], data_dict['pos_y'])
    print('Total distance: {:.2f} m'.format(distance))

    # intervention percentage
    intervene = calculate_intervene(data_dict['flag'])
    print('Intervention (%): {:.2f} %'.format(intervene*100))

    # invervention per meter
    intervene_per_meter = calculat_intervene_per_distance(data_dict['flag'], distance)
    print('Intervention per meter: {:.2f}'.format(intervene_per_meter))


if __name__ == '__main__':
    # Dataset folder
    # dataset_dir = '/media/lab/Hard Disk/' + 'my_datasets'
    dataset_dir = 'D:/Github/neptune/my_datasets'
    # Subject list
    subject_list = [
        'subject1',
        # 'subject2',
        # 'subject3',
        # 'subject4',
        # 'subject5',
        # 'subject6',
        # 'subject7',
        # 'subject8',
        # 'subject9',
        # 'subject10',
        # 'subject11',
        # 'subject12',
        # 'subject13',
        # 'subject14',
    ]
    # Map list
    map_list = [
        'map1',
        'map2',
        'map3',
        'map4',
        'map5',
        'map7',
        'map8',
        # 'map9',
        # 'o1',
        'o2',
    ]
    # Iteration
    iteration = 'iter' + str(0)

    # # Init the plot
    # fig, axes = plt.subplots(2, 5, sharex=False, sharey=False)

    # Start the loop
    for subject in subject_list:
        for map in os.listdir(os.path.join(dataset_dir, subject)):
            if map in map_list:
                folder_path = os.path.join(dataset_dir, subject, map, iteration)
                if os.path.isdir(folder_path):
                    for subfolder in os.listdir(folder_path):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        data_dict = read_data(subfolder_path)
                        print(subfolder_path)
                        # Process the data
                        process_data(data_dict)
                        print('******************')