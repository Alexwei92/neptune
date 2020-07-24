import numpy as np
import rosbag
import pathlib as plb

import subprocess, yaml

filename = 'ccBag.bag'

info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', filename], stdout=subprocess.PIPE).communicate()[0])

print(info_dict)

bag = rosbag.bag(filename)