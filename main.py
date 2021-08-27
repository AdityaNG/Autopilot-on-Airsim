import airsim
import cv2
import numpy as np
import os
import time
import math
import pprint
import argparse
import threading
import plotter
import subprocess
from autopilot_utils import *
from datetime import datetime
import signal
import matplotlib.pyplot as plt
from multiprocessing import Process, Array, Pool, Queue
import ctypes

IMAGE_SHAPE = (144,256,3)

point_cloud_array = Queue()

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera_list', nargs='+', default=['0', '1', '2', '3', '4'], help='List of cameras visualised : [0, 1, ... , 4]')
parser.add_argument('-v', '--view_list', nargs='+', default=['0', '3', '5'], help='List of cameras visualised : [0, 1, ... , 6]')
parser.add_argument('-exe', '--executable', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Apps/AirSimNH_1.4.0/LinuxNoEditor/AirSimNH.sh")), help='Path to Airshim.sh')
parser.add_argument('-s', '--settings', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Autopilot/settings.json")), help='Path to Airshim settings.json')
args = parser.parse_args()

# Logging the data to disk
current_date = str(datetime.now().strftime("%d-%m-%Y_%H%M%S"))
logfilename = 'logs/{}_LOG.txt'.format(current_date)
simfilename = 'logs/{}_SIM.txt'.format(current_date)
if not os.path.exists(os.path.dirname(logfilename)):
    try:
        os.makedirs(os.path.dirname(logfilename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Open sim log file
simfile = open(simfilename, 'w')
# Launch the sim process
proc = subprocess.Popen([args.executable, '-WINDOWED', '-ResX=640', '-ResY=480', '--settings', args.settings], stdout=simfile)

# Wait for Airsim to launch
# TODO : Replace wait with polling
time.sleep(5)

        
def get_image(req, mode_name, camera_data):
    """
       get_image -> connects to server to get ONE camera's frame
       Is a seperate process
       Called with themultiprocessing.Pool as a batch
    """
    global client
    responses = client.simGetImages(requests=[req])
    response = responses[0]
    if response.pixels_as_float:
        depth = np.array(response.image_data_float, dtype=np.float32)
        depth = depth.reshape(response.height, response.width)
        img = np.array(depth * 255, dtype=np.uint8)

    else:
        img = response.image_data_uint8
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)

    if mode_name[response.image_type] == 'DepthVis':
        p_mat = np.array(camera_data[response.camera_name].proj_mat.matrix)
        Quat = camera_data[response.camera_name].pose.orientation
        Quat = [Quat.w_val, Quat.x_val, Quat.y_val, Quat.z_val ]
        
        R_mat = np.array(quaternion_rotation_matrix(Quat))
        T_mat = camera_data[response.camera_name].pose.position
        T_mat = np.array([T_mat.x_val, T_mat.y_val, T_mat.z_val] )
        points = cv2.reprojectImageTo3D(depth, p_mat)
        points = cv2.reprojectImageTo3D(img, p_mat)
        points = np.array([p for r in points for p in r])

        points_rt = np.zeros(shape=points.shape)
        for i in range(len(points)):
            p = points[i]
            points_rt[i] = R_mat.dot(p) + T_mat

        return points_rt, img, response.camera_name, response.image_type
    return None, img, response.camera_name, response.image_type


def setup():
    """
        setup for each get_image subprocess
        Each get_image subprocess gets its own instance of the airsim.CarClient()
        Note : airsim.CarClient() is not thread safe
    """
    global client
    # Connect to Airsim
    client = airsim.CarClient()
    client.confirmConnection()


def image_loop(point_cloud_array):
    """
        image_loop is launched as a subprocess
        point_cloud_array is a multiprocessing.Queue() object
        The new point cloud gets pushed onto the Queue
    """
    white_bg = np.zeros((144,256,3))
    pp = pprint.PrettyPrinter(indent=4)
    client = airsim.CarClient()
    client.reset()
    client.confirmConnection()
    client.enableApiControl(False)
    print("API Control enabled: %s" % client.isApiControlEnabled())
    car_controls = airsim.CarControls()

    camera_data = {}

    for camera_name in range(5):
        camera_info = client.simGetCameraInfo(str(camera_name))
        camera_data[str(camera_name)] = camera_info
        #print("CameraInfo %d:" % camera_name)
        #pp.pprint(camera_info)

    cam_name = {
            '0': 'Front',
            '1': 'Back',
            '2': 'Right',
            '3': 'Left',
            '4': 'FrontLR',
    }
    mode_name = {
            0: 'Scene', 
            1: 'DepthPlanner', 
            2: 'DepthPerspective',
            3: 'DepthVis', 
            4: 'DisparityNormalized',
            5: 'Segmentation',
            6: 'SurfaceNormals'
    }

    reqs = [] # List of requests for images
    axi = {}  # dict of subplots
    for ind in range(len(args.camera_list)):
        i = args.camera_list[ind]
        axi.setdefault(i, {})
        for j, v in enumerate(args.view_list):
            m = mode_name[int(v)]
            #axi[i].setdefault(m, {})
            axi[i][int(v)] = plt.subplot(len(args.camera_list), len(args.view_list), len(args.view_list)*ind +j+1)    
            axi[i][int(v)].title.set_text(cam_name[i] + '_' + m)
            as_float = False
            if m == 'DepthVis':
                as_float = True
            reqs.append(airsim.ImageRequest(i, int(v), as_float, False))
    
    # Argument list for each get_image
    args_list = []
    for r in reqs:
        args_list.append((r, mode_name, camera_data))
    
    with Pool(initializer=setup, initargs=[]) as pool:
        
        # Async plotting
        plt.ion()
        plt.show(block=False)

        while True:
            #for cam in axi: for type in axi[cam]: axi[cam][type].imshow(white_bg)
            results = pool.starmap(get_image, args_list)
            final_points = np.array([[0, 0, 0], ])
            for points_rt, img, camera_name, image_type in results:
                if type(points_rt) == np.ndarray:
                    final_points = np.concatenate((final_points, points_rt), )

                if camera_name in args.camera_list:
                    axi[camera_name][int(image_type)].imshow(img)
                    
            plt.pause(0.001)
            # Send the points to the queue
            point_cloud_array.put(final_points)


# Process to call images from the sim and process them to generate point_cloud_array
p = Process(target=image_loop, args=(point_cloud_array, ))
p.start()

# Start blocking start_graph call
plotter.start_graph(point_cloud_array)

# Once graph window is closed, kill the image_loop process
# TODO : Replace kill with a gracefull end
p.kill()
proc.kill()