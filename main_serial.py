from glob import glob
import os
import sys
sys.path.append("/home/aditya/AirSim/PythonClient/")

import airsim
import cv2
import numpy as np

import time
import math
import pprint
import argparse
import threading

import subprocess
from autopilot_utils import *
from datetime import datetime
import signal
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from multiprocessing import Process, Array, Pool, Queue
import ctypes

from stereo_vision import stereo_vision
#from manydepth  import manydepth
#from monodepth2  import monodepth2

import generate_cameras

IMAGE_SHAPE = (144,256,3)


point_cloud_array = Queue()

parser = argparse.ArgumentParser()
#parser.add_argument('-c', '--camera_list', nargs='+', default=['0', ], help='List of cameras visualised : [0, 1]')
parser.add_argument('-c', '--camera_list', nargs='+', default=list(map(str, list(range(1, generate_cameras.NUM_CAMS+1)) )), help='List of cameras visualised : [0, 1]')
#parser.add_argument('-v', '--view_list', nargs='+', default=['0', '4', '7'], help='List of cameras visualised : [0, 1, ... , 6]')
parser.add_argument('-v', '--view_list', nargs='+', default=['0', '4'], help='List of cameras visualised : [0, 1, ... , 6]')
#parser.add_argument('-exe', '--executable', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Apps/AirSimNH_1.4.0/LinuxNoEditor/AirSimNH.sh")), help='Path to Airshim.sh')
parser.add_argument('-exe', '--executable', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Apps/AirSimNH_1.6.0/LinuxNoEditor/AirSimNH.sh")), help='Path to Airshim.sh')
parser.add_argument('-s', '--settings', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Autopilot/settings.stereo.json")), help='Path to Airshim settings.stereo.json')
parser.add_argument('-p3', '--plot_3D', action='store_true', help='3D plotting')

parser.add_argument('-n', '--no_exec', action='store_true', help='Flag to execute ONLY the simulator')

args = parser.parse_args()

# Logging the data to disk
current_date = str(datetime.now().strftime("%d-%m-%Y_%H%M%S"))
logfilename = 'logs/{}_LOG.txt'.format(current_date)
simfilename = 'logs/{}_SIM.txt'.format(current_date)
simfile_err_name = 'logs/{}_SIM_ERR.txt'.format(current_date)
if not os.path.exists(os.path.dirname(logfilename)):
    try:
        os.makedirs(os.path.dirname(logfilename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Open sim log file
simfile = open(simfilename, 'w')
simfile_err = open(simfile_err_name, 'w')
# Launch the sim process
airsim_proc = subprocess.Popen([args.executable, '-WINDOWED', '-ResX=640', '-ResY=480', '--settings', args.settings], stdout=simfile, stderr=simfile_err)

global client
connection_failed = True
while connection_failed:
    try:
        client = airsim.CarClient()
        client.confirmConnection()
        connection_failed = False
        print(". done", end="")
    except:
        connection_failed = True
        time.sleep(1)
# Wait for Airsim to launch
# TODO : Replace wait with polling
#time.sleep(5)

        
def get_image(req, mode_name, camera_data):
    """
       get_image -> connects to server to get ONE camera's frame
       Is a seperate process
       Called with themultiprocessing.Pool as a batch
    """
    global client
    points_rt = None
    if type(req)!=list:
        req = [req,]
    responses = client.simGetImages(requests=req)
    print("responses got")

    FIN_res = []

    for response in responses:
        p_mat = np.array(camera_data[response.camera_name].proj_mat.matrix)
        Quat = camera_data[response.camera_name].pose.orientation
        Quat = [Quat.w_val, Quat.x_val, Quat.y_val, Quat.z_val ]
            
        R_mat = np.array(quaternion_rotation_matrix(Quat))
        T_mat = camera_data[response.camera_name].pose.position
        T_mat = np.array([T_mat.x_val, T_mat.y_val, T_mat.z_val] )

        if response.pixels_as_float:
            print("response.pixels_as_float")
            depth = np.array(response.image_data_float, dtype=np.float32)
            depth = depth.reshape(response.height, response.width)
            #img = np.array(depth * 255, dtype=np.uint8)
            img = np.array(depth * 255, dtype=np.float32)

        else:
            img = response.image_data_uint8
            img = np.frombuffer(img, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 3)

        #if mode_name[response.image_type] == 'DepthVis':
        if False and mode_name[response.image_type] == 'DisparityNormalized':

            p_mat_2 = np.array([
                [959.779968, 0.000000, 959.290331, 0.000000],
                [0.000000, 959.867798, 539.535675, 0.000000],
                [0.000000, 0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 0.000000, 1.000000]
            ])
            
            #points = cv2.reprojectImageTo3D(depth, p_mat)
            points = cv2.reprojectImageTo3D(img, p_mat)
            points_rt = np.array([p for r in points for p in r])

            """
            points_rt = np.zeros(shape=points.shape)
            for i in range(len(points)):
                p = points[i]
                points_rt[i] = R_mat.dot(p) + T_mat
            # """

        """
        elif mode_name[response.image_type] == 'Scene':
            if prev_frame != type(None):
                depth = md.eval(img, prev_frame)
                
                #points = cv2.reprojectImageTo3D(depth, p_mat)
                points = cv2.reprojectImageTo3D(img, p_mat)
                points = np.array([p for r in points for p in r])

                points_rt = np.zeros(shape=points.shape)
                for i in range(len(points)):
                    p = points[i]
                    points_rt[i] = R_mat.dot(p) + T_mat
            prev_frame = img.copy()
        """

        FIN_res.append((points_rt, img, response.camera_name, response.image_type))

    return FIN_res


def setup():
    """
        setup for each get_image subprocess
        Each get_image subprocess gets its own instance of the airsim.CarClient()
        Note : airsim.CarClient() is not thread safe
    """
    global client
    # Connect to Airsim
    client = airsim.CarClient()


    print("Connected Proceess : ", os.getpid())


def image_loop(point_cloud_array):
    """
        image_loop is launched as a subprocess
        point_cloud_array is a multiprocessing.Queue() object
        The new point cloud gets pushed onto the Queue
    """
    global sv
    # Stereo vision object
    # sv = stereo_vision(width=480, height=270, defaultCalibFile=False, CAMERA_CALIBRATION_YAML="calibration/fsds.yml", objectTracking=False, display=True, graphics=False, scale=1, pc_extrapolation=False)


    white_bg = np.zeros((144,256,3))
    pp = pprint.PrettyPrinter(indent=4)
    client = airsim.CarClient()
    client.reset()
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
        '1': 'FrontL',
        str(generate_cameras.NUM_CAMS): 'FrontR'
    }
    for i in range(2, generate_cameras.NUM_CAMS):
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

    reqs = [] # List of requests for images
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
            as_float = False
            #if m=='DepthVis' or m=='DisparityNormalized' or m=='DepthPlanar' or m=='DepthPerspective':
            if m=='DepthVis' or m=='DisparityNormalized':
                as_float = True
                as_float = False
            
            reqs.append(airsim.ImageRequest(i, int(v), as_float, False))
        #axi[i]['NN_Depth'] = plt.subplot(plots_height, plots_width, plots_width*ind +j+2)    
        #axi[i]['NN_Depth'].title.set_text(cam_name[i] + '_NN_depth')

        if '0' not in args.view_list:
            reqs.append(airsim.ImageRequest(i, 0, False, False))

        #axi[i][plots_width+2] = plt.subplot(plots_height, plots_width, plots_width*ind +j+3)    
        #axi[i][plots_width+2].title.set_text(cam_name[i] + '_Monodepth2')
    # Argument list for each get_image
    args_list = []

    #for r in reqs: args_list.append((r, mode_name, camera_data))

    args_list = [(reqs, mode_name, camera_data), ]
    
    with Pool(initializer=setup, initargs=[], processes=1) as pool:
        
        # Async plotting
        plt.ion()
        plt.show(block=False)

        prev_frame = {}

        while True:
            #for cam in axi: for type in axi[cam]: axi[cam][type].imshow(white_bg)
            print("Before pool")
            results = pool.starmap(get_image, args_list)
            print("After pool")
            final_points = np.array([[0, 0, 0], ])

            for c_res in results:
                for points_rt, img, camera_name, image_type in c_res:
                    if type(points_rt) == np.ndarray:
                        print("points_rt")
                        print(points_rt)
                        final_points = np.concatenate((final_points, points_rt), )

                    if camera_name in args.camera_list:
                        if str(image_type) in args.view_list:
                            if mode_name[image_type] == 'DisparityNormalized':
                                #axi[camera_name][int(image_type)].imshow(img, cmap='viridis')
                                disp_resized_np = img
                                vmax = np.percentile(disp_resized_np, 95)
                                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
                                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                                #axi[camera_name][int(image_type)].imshow(colormapped_im, cmap='plasma_r')
                                axi[camera_name][int(image_type)].imshow(colormapped_im)
                            elif mode_name[image_type] == 'Infrared':
                                print('Infrared', img.shape)
                                if len(img.shape) != 2:
                                    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                else:
                                    img_grey = img
                                print(img_grey.shape)
                                axi[camera_name][int(image_type)].imshow(img_grey, cmap='magma')
                            else:
                                axi[camera_name][int(image_type)].imshow(img)
                        
                        if mode_name[image_type] == 'Scene':
                            # RBD image

                            # TODO: Save images to disk
                            print('Scene', img.shape)

                            prev_frame.setdefault(camera_name, None)
                            prev_frame[camera_name] = img
                        

                plt.pause(0.001)
                
                # Send the points to the queue
                #point_cloud_array.put(points_sv)
                point_cloud_array.put(final_points)


try:
    # Process to call images from the sim and process them to generate point_cloud_array
    image_loop_proc = Process(target=image_loop, args=(point_cloud_array, ))
    if not args.no_exec:
        image_loop_proc.start()
        if args.plot_3D:
            # Start blocking start_graph call
            import plotter
            plotter.start_graph(point_cloud_array)
        else:
            input("Press enter to quit")
    else:
        input("Press enter to quit")

    # Once graph window is closed, kill the image_loop process
except e:
    print(e)
finally:
    os.killpg(os.getpgid(airsim_proc.pid), signal.SIGTERM)  # Send the signal to all the process groups
    os.killpg(os.getpgid(image_loop_proc.pid), signal.SIGTERM)