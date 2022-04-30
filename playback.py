import os
import sys
sys.path.append("/home/aditya/AirSim/PythonClient/")

import cv2
import numpy as np

import argparse

from autopilot_utils import *
from datetime import datetime
import signal
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import generate_cameras

import airsim

import PIL


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-04-30-10-00-20/")), help='Path to Airshim recording folder')
parser.add_argument('-c', '--camera_list', nargs='+', default=list(map(str, list(range(0, generate_cameras.NUM_CAMS+1)) )), help='List of cameras visualised : [0, 1]')
parser.add_argument('-v', '--view_list', nargs='+', default=['0', '4'], help='List of cameras visualised : [0, 1, ... , 6]')

args = parser.parse_args()

df_path = os.path.join(args.recording_path, 'airsim_rec.txt')

df = pd.read_csv(df_path, sep='\t')
df.set_index('TimeStamp')
#print(df)

cam_name = {
    '0': 'FrontL',
    str(generate_cameras.NUM_CAMS): 'FrontR'
}
for i in range(1, generate_cameras.NUM_CAMS):
    cam_name[str(i)] = 'C' + str(i)
mode_name = {
    0: 'Scene', 
    1: 'DepthPlanar', 
    2: 'DepthPerspective',
    3: 'DepthVis', 
    4: 'DisparityNormalized',
    5: 'Segmentation',
    6: 'SurfaceNormals',
    7: 'Infrared'
}

plt.ion()
plt.show(block=False)

axi = {}  # dict of subplots
plots_height = len(args.camera_list)
#plots_width = len(args.view_list) + 1
plots_width = len(args.view_list)
for ind in range(len(args.camera_list)):
    i = args.camera_list[ind]
    axi.setdefault(i, {})
    for j, v in enumerate(args.view_list):
        m = mode_name[int(v)]
        #axi[i].setdefault(m, {})
        axi[i][int(v)] = plt.subplot(plots_height, plots_width, plots_width*ind +j+1)    
        axi[i][int(v)].title.set_text(cam_name[i] + '_' + m)

for i, row in df.iterrows():
    files = row['ImageFile'].split(";")
    files_path = list(map(lambda x: os.path.join(args.recording_path, 'images', x), files))
    #print(files)
    for i, f in enumerate(files_path):
        cam_id, img_format = files[i].split("_")[2:4]
        img_format = int(img_format)
        print(cam_id, img_format)
        if f.endswith('.ppm'):
            img = PIL.Image.open(f)
            print(img.size)
            img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
        elif f.endswith('.pfm'):
            img, scale = airsim.read_pfm(f)
        else:
            print("Unknown format")

        #axi[cam_id][img_format].imshow(img, cmap='magma')
        axi[cam_id][img_format].imshow(img)
    plt.pause(0.001)
