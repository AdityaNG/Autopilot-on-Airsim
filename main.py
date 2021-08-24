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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera_list', nargs='+', default=['0', '1', '2', '3', '4'], help='List of cameras visualised : [0, 1, ... , 4]')
parser.add_argument('-exe', '--executable', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Apps/AirSimNH_1.4.0/LinuxNoEditor/AirSimNH.sh")), help='Path to Airshim.sh')
parser.add_argument('-s', '--settings', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Autopilot/settings.json")), help='Path to Airshim settings.json')
args = parser.parse_args()

pp = pprint.PrettyPrinter(indent=4)

current_date = str(datetime.now().strftime("%d-%m-%Y_%H%M%S"))
logfilename = 'logs/{}_LOG.txt'.format(current_date)
simfilename = 'logs/{}_SIM.txt'.format(current_date)
if not os.path.exists(os.path.dirname(logfilename)):
    try:
        os.makedirs(os.path.dirname(logfilename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
simfile = open(simfilename, 'w')
proc = subprocess.Popen([args.executable, '-WINDOWED', '-ResX=640', '-ResY=480', '--settings', args.settings], stdout=simfile)
time.sleep(5)
#plotter.init_disp()

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
    print("CameraInfo %d:" % camera_name)
    pp.pprint(camera_info)

cam_name = {
        '0': 'Front',
        '1': 'Back',
        '2': 'Right',
        '3': 'Left',
        '4': 'FrontLR',
}
cam_name_show = args.camera_list
mode_name = {
        0: 'Scene', 
        1: 'DepthPlanner', 
        2: 'DepthPerspective',
        3: 'DepthVis', 
        4: 'DisparityNormalized',
        5: 'Segmentation',
        6: 'SurfaceNormals'
}

def image_loop():
    while True:
        responses = client.simGetImages([
            #airsim.ImageRequest("0", airsim.ImageType.DepthVis),
            #airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True),
            #airsim.ImageRequest("2", airsim.ImageType.Segmentation),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("4", airsim.ImageType.Scene, False, False),

            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
            airsim.ImageRequest("1", airsim.ImageType.DepthVis, True, False),
            airsim.ImageRequest("2", airsim.ImageType.DepthVis, True, False),
            airsim.ImageRequest("3", airsim.ImageType.DepthVis, True, False),
            #airsim.ImageRequest("4", airsim.ImageType.DepthVis, True, False),

            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("1", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("2", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("3", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("4", airsim.ImageType.Segmentation, False, False),
            #airsim.ImageRequest("4", airsim.ImageType.Scene, False, False),
            #airsim.ImageRequest("4", airsim.ImageType.DisparityNormalized),
            #airsim.ImageRequest("4", airsim.ImageType.SurfaceNormals)
            ])

        final_points = np.array([[0, 0, 0], ])
        for i, response in enumerate(responses):
            #print(dir(response))
            #filename = os.path.join(tmp_dir, str(x) + "_" + str(i))
            if response.pixels_as_float:
                #print("pixels_as_float Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                #airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
                #img = airsim.get_pfm_array(response)
                depth = np.array(response.image_data_float, dtype=np.float32)
                depth = depth.reshape(response.height, response.width)
                img = np.array(depth * 255, dtype=np.uint8)

            else:
                #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                #airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
                img = response.image_data_uint8
                img = np.frombuffer(img, dtype=np.uint8)
                #img = np.fromstring(img, dtype=np.uint8)
                img = img.reshape(response.height, response.width, 3)
            #print(img)
            if mode_name[response.image_type] == 'DepthVis':
                p_mat = np.array(camera_data[response.camera_name].proj_mat.matrix)
                #R_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                #T_mat = np.array([0, 0, 0])
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
                #points = [(0,0,0), ]
                #points = list(filter(lambda p: (-float('inf')<p).all() and (p<float('inf')).all(), points))
                #points = list(filter(lambda p: not p[0].isinf() and not p[0].isinf() and not p[0].isinf(), points))
                print(points_rt.shape)
                final_points = np.concatenate((final_points, points_rt))

            if response.camera_name in cam_name_show:
                #cv2.imshow(cam_name[response.camera_name] + "_" + mode_name[response.image_type], img)
                #cv2.waitKey(1)
                pass
        plotter.plot_points(final_points)

image_loop_thread = threading.Thread(target=image_loop, daemon=True)
image_loop_thread.start()
plotter.start_graph()
while proc.poll() is not None:
    pass

print("Sim shutdown")
# process has not (yet) terminated. 

# Try to stop it gracefully.
#os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
os.kill(proc.pid, signal.SIGINT)
#time.sleep(5)

# Still running?
if proc.poll() is None:
    # Kill it with fire
    proc.terminate()
    # Wait for it to finish
    proc.wait()

# Going to kill all related processes created by simulation_process
#os.system("taskkill /im Blocks* /F")
os.system("killall AirSimNH")
exit()
while True:
    #image_loop()
    try:
        plotter.main_loop()
        plotter.start_graph()
    except:
        pass
