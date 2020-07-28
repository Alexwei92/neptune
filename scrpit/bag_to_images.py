#!/usr/bin/env python

"""Extract images and telemetry data from a ROS bag.
"""

import os
import argparse
import rosbag
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import shutil, csv, string
import __future__


image_topics = ['/camera/color/image_raw',
                '/camera/depth/image_rect_raw',
]

telemetry_topics = ['/mavros/state',
                    '/mavros/rc/in',
                    '/mavros/local_position/velocity_body',
                    '/mavros/global_position/rel_alt',
                    '/mavros/global_position/local',
                    '/camera/color/image_raw',
                    '/camera/depth/image_rect_raw'
]

# write image data
def write_image(bagfile, path_color, path_depth, path):
    bridge = CvBridge()

    count_color, count_color_bad = 0, 0
    count_depth, count_depth_bad = 0, 0


    f1 = open(path+'/'+'color_info.csv','w')
    f2 = open(path+'/'+'depth_info.csv','w')
    filewriter1 = csv.writer(f1, delimiter = ',')
    filewriter2 = csv.writer(f2, delimiter = ',')
    filewriter1.writerow(['rosbagTimestamp', 'flag'])
    filewriter2.writerow(['rosbagTimestamp', 'flag'])
    
    for topic, msg, t in bagfile.read_messages(topics=image_topics):
        values = [str(t)]
        if topic == '/camera/color/image_raw':
            try:
                cv_img_color = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") # not rgb8
                #print(msg.encoding)
                cv2.imwrite(os.path.join(path_color, "frame%06i.jpg" % count_color), cv_img_color)
                count_color += 1
                values.append(1)
            except:
                count_color_bad += 1
                values.append(0)
                pass
            filewriter1.writerow(values)
        else:
            try:
                cv_img_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1") # 8UC1?
                #print(msg.encoding)
                cv2.imwrite(os.path.join(path_depth, "frame%06i.jpg" % count_depth), cv_img_depth)
                count_depth += 1
                values.append(1)
            except:
                count_depth_bad += 1
                values.append(0)
                pass            
            filewriter2.writerow(values)

    print("Wrote %i Color images and %i Depth images." % (count_color, count_depth))
    if count_depth_bad+count_color_bad > 0:
        print("[Warning: Found %i broken Color images and %i broken Depth images.]" % (count_color_bad, count_depth_bad))

    f1.close()
    f2.close()
    return


# write telemetry data
def write_telemetry(bagfile, path):
    for topic_name in telemetry_topics:
        count = 0
        filename = path+'/'+string.replace(topic_name, '/', '_')+'.csv'
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            firstIteration = True #allows header row
            for subtopic, msg, t in bagfile.read_messages(topics=topic_name):
                #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
				#	- put it in the form of a list of 2-element lists
                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):	#should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)
                #write the first row from the first element of each pair
                if firstIteration:	# header
                    headers = ["rosbagTimestamp"]	#first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    filewriter.writerow(headers)
                    firstIteration = False
                # write the value from each pair to the file
                values = [str(t)]	#first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair) > 1:
                        values.append(pair[1])
                filewriter.writerow(values)
                count += 1
    return

if __name__ == '__main__':
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images and telemetry data from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    #parser.add_argument("output_dir", help="Output directory.")
    #parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    try:
        bag = rosbag.Bag(args.bag_file, "r")
        print("Extract images from '%s'" % (args.bag_file))
    except IOError:
        print("[Error: No such file: '%s']\nFailed!" % (args.bag_file))
        exit(1)

    # Create the output directory
    path = os.path.join('./', args.bag_file[:-4])
    path_color = path+'/'+'color'
    path_depth = path+'/'+'depth'
    if not os.path.isdir(path_color):
        os.makedirs(path_color)
    if not os.path.isdir(path_depth):
        os.makedirs(path_depth)
    shutil.copyfile(args.bag_file, path+'/'+bag.filename)

    # Write color and depth images
    try:
        write_image(bag, path_color, path_depth, path)
        write_telemetry(bag, path)
        print("Successful.")
    finally:
        bag.close()
    