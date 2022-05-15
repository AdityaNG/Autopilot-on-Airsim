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

from multiprocessing import Process, Array, Pool, Queue
import subprocess

parser = argparse.ArgumentParser()
#parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-04-30-10-00-20/")), help='Path to Airshim recording folder')
#parser.add_argument('-v', '--view_list', nargs='+', default=['0', '4'], help='List of cameras visualised : [0, 1, ... , 6]')

#parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-05-15-12-40-30/")), help='Path to Airshim recording folder')
#parser.add_argument('-v', '--view_list', nargs='+', default=['0', '2'], help='List of cameras visualised : [0, 1, ... , 6]')

#parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-05-15-12-55-55/")), help='Path to Airshim recording folder')
#parser.add_argument('-v', '--view_list', nargs='+', default=['0', '1'], help='List of cameras visualised : [0, 1, ... , 6]')

#parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-05-15-14-02-08/")), help='Path to Airshim recording folder')
parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-05-15-14-08-50/")), help='Path to Airshim recording folder')
parser.add_argument('-v', '--view_list', nargs='+', default=['0', '1', '2', '4', '5', '7'], help='List of cameras visualised : [0, 1, ... , 6]')
parser.add_argument('-c', '--camera_list', nargs='+', default=list(map(str, list(range(0, generate_cameras.NUM_CAMS+1)) )), help='List of cameras visualised : [0, 1]')
parser.add_argument('-p3', '--plot_3D', action='store_true', help='3D plotting')

args = parser.parse_args()

df_path = os.path.join(args.recording_path, 'airsim_rec.txt')

df = pd.read_csv(df_path, sep='\t')
df.set_index('TimeStamp')
IMAGE_SHAPE = (144,256,3)


point_cloud_array = Queue()

def compute_points(depth_map):
    print("compute points")
    p_mat = np.array([
        [959.779968, 0.000000, 959.290331, 0.000000],
        [0.000000, 959.867798, 539.535675, 0.000000],
        [0.000000, 0.000000, 1.000000, 0.000000],
        [0.000000, 0.000000, 0.000000, 1.000000]
    ])
    
    p_mat_alt = np.array([
        [-0.501202762*2, 0.000000000, 0.000000000, 0.000000000],
        [0.000000000, -0.501202762*2, 0.000000000, 0.000000000],
        [0.000000000, 0.000000000, 10.00000000*2, 100.00000000],
        [0.000000000, 0.000000000, -10.0000000, 0.000000000*2]
    ]) 

    p_mat_git = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
        [0.000000000, -0.501202762, 0.000000000, 0.000000000],
        [0.000000000, 0.000000000, 10.00000000, 100.00000000],
        [0.000000000, 0.000000000, -10.0000000, 0.000000000]
    ]) 

    p_mat_sim = np.array([
        [ 0.,          0.57735026,  0.,          0.        ],
        [ 0.,          0.,         -1.02640045,  0.        ],
        [ 0.,          0.,          0.,         10.        ],
        [-1.,          0.,          0.,          0.        ]
    ])

    p_mat = p_mat_git
    #p_mat = (p_mat_sim + p_mat_git) /2

    #depth_map = np.uint8(depth_map)
    #depth_map = depth_map.astype(np.uint8)
    print(depth_map.shape)
    print(depth_map.dtype)
    #info = np.info(depth_map.dtype) # Get the information of the incoming image type
    #depth_map = np.uint8(depth_map / info.max)
    #depth_map = np.uint8(depth_map)
    depth_map = np.float32(depth_map) # CV_32FC1
    #depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

            
    #points = cv2.reprojectImageTo3D(depth, p_mat)
    points = cv2.reprojectImageTo3D(depth_map, p_mat)
    points_rt = np.array([p for r in points for p in r])
    points_rt = points_rt / 500
    return points_rt

def image_loop(point_cloud_array):
    """
        image_loop is launched as a subprocess
        point_cloud_array is a multiprocessing.Queue() object
        The new point cloud gets pushed onto the Queue
    """
    points = np.array([
        [1,1,1],
        [2,2,2]
    ])

    cam_name = {
        '0': 'FrontL',
        str(generate_cameras.NUM_CAMS): 'FrontR'
    }
    #for i in range(1, generate_cameras.NUM_CAMS):
    for i in args.camera_list:
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
            #axi[i][int(v)] = plt.subplot(plots_height, plots_width, plots_width*ind +j+1)    
            axi[i][int(v)] = plt.subplot(plots_width, plots_height, plots_height*j +ind+1)    
            axi[i][int(v)].title.set_text(cam_name[i] + '_' + m)
    
    for i, row in df.iterrows():
        final_points = np.array([[0, 0, 0], ])
        files = row['ImageFile'].split(";")
        files_path = list(map(lambda x: os.path.join(args.recording_path, 'images', x), files))
        #print(files)
        for i, f in enumerate(files_path):
            cam_id, img_format = files[i].split("_")[2:4]
            img_format = int(img_format)
            if f.endswith('.ppm'):
                img = PIL.Image.open(f)
                img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            elif f.endswith('.pfm'):
                img, scale = airsim.read_pfm(f)
            else:
                print("Unknown format")

            print(i, img_format)
            if cam_id=='0' and img_format==4:
                points_rt = compute_points(img)
                final_points = np.concatenate((final_points, points_rt), )

            #axi[cam_id][img_format].imshow(img, cmap='magma')
            if cam_id in args.camera_list and str(img_format) in args.view_list:
                axi[cam_id][img_format].imshow(img)
        point_cloud_array.put(final_points)
        plt.pause(0.001)
        


try:
    # Process to call images from the sim and process them to generate point_cloud_array
    image_loop_proc = Process(target=image_loop, args=(point_cloud_array, ))
    
    image_loop_proc.start()
    if args.plot_3D:
        # Start blocking start_graph call
        import plotter
        plotter.start_graph(point_cloud_array)
    else:
        input("Press enter to quit")

    # Once graph window is closed, kill the image_loop process
except Exception as e:
    print(e)
finally:
    os.killpg(os.getpgid(image_loop_proc.pid), signal.SIGTERM)