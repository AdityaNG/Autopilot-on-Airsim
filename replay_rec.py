import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt

recordings_folder_path = os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim"))

recordings_list = list(filter(lambda x: x!='settings.json', os.listdir(recordings_folder_path) ))
recordings_list.sort()

recent_recording = ''
if len(recordings_list)>0:
    recent_recording = os.path.join(recordings_folder_path, recordings_list[-1])


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--recording', type=str, default=recent_recording, help='Path to recording')
args = parser.parse_args()

images_dir = os.path.join(args.recording, 'images')
data_file =  os.path.join(args.recording, 'airsim_rec.txt')
print(images_dir)

axi = []
axi.append(plt.subplot(3, 1, 1))
axi.append(plt.subplot(3, 1, 2))
axi.append(plt.subplot(3, 1, 3))

with open(data_file) as fp:
    headers = fp.readline().split('\t')
    print(headers)
    while True:
        line = fp.readline()
 
        if not line:
            break
        
        VehicleName, TimeStamp, POS_X, POS_Y, POS_Z, Q_W, Q_X, Q_Y, Q_Z, Throttle, Steering, Brake, Gear, Handbrake, RPM, Speed, ImageFile = line.split('\t')

        for ind, i in enumerate(ImageFile.split(';')):
            i_path = os.path.join(args.recording, 'images', i).replace('\n','')
            print(i_path)
            img = cv2.imread(i_path)
            #if type(img) != type(None):
            #print(img.dtype)
            axi[ind].imshow(img)
        
        plt.pause(0.0001)