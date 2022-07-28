import os
import sys
sys.path.append("/home/aditya/AirSim/PythonClient/")

import cv2
import numpy as np

import argparse

#from autopilot_utils import *
from datetime import datetime
import signal
import pandas as pd
import airsim_utils.generate_cameras as generate_cameras

import airsim

import PIL
from PIL import Image

from multiprocessing import Process, Queue
import subprocess
from scipy.spatial.transform import Rotation

from helper import cv2_grid_display

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--recording_path', type=str, default=os.path.abspath(os.path.join(os.getenv("HOME"), "Documents/AirSim/2022-05-22-11-10-49/")), help='Path to Airshim recording folder')
parser.add_argument('-s', '--settings_path', type=str, default=os.path.abspath(os.path.join(os.getcwd(), "airsim_settings", "settings.multicam.json")), help='Path to Airshim recording folder')
parser.add_argument('-v', '--view_list', nargs='+', default=['0', '1', '2', '4', '5', '7'], help='List of cameras visualised : [0, 1, ... , 6]')
parser.add_argument('-c', '--camera_list', nargs='+', default=list(map(str, list(range(0, generate_cameras.NUM_CAMS+1)) )), help='List of cameras visualised : [0, 1]')
parser.add_argument('-p3', '--plot_3D', action='store_true', help='3D plotting')
parser.add_argument('-w', '--wait', action='store_true', help='Wait for keypress to play')

args = parser.parse_args()

df_path = os.path.join(args.recording_path, 'airsim_rec.txt')

df = pd.read_csv(df_path, sep='\t')
df.set_index('TimeStamp')
IMAGE_SHAPE = (144,256,3)

import json

settings_file = open(args.settings_path, "r")
settings_str = settings_file.read()
settings_file.close()
settings_json = json.loads(settings_str)

camera_details = settings_json['Vehicles']['PhysXCar']['Cameras']
for cam_id in camera_details:
	print(cam_id, camera_details[cam_id])

#exit()
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
		[ 0.,		  0.57735026,  0.,		  0.		],
		[ 0.,		  0.,		 -1.02640045,  0.		],
		[ 0.,		  0.,		  0.,		 10.		],
		[-1.,		  0.,		  0.,		  0.		]
	])

	p_mat_git_2 = np.array([
		[959.779968,	0.000000,   959.290331, 0.000000],
		[0.000000,	  959.867798, 539.535675, 0.000000],
		[0.000000,	  0.000000,   1.000000,   0.000000],
		[0.,		   0.,		 0.,		  2.		]
	]) / 4000

	

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
	transf = np.hstack((
		np.ones(shape=(points_rt.shape[0], 1)) * -1,
		np.ones(shape=(points_rt.shape[0], 1)),
		np.ones(shape=(points_rt.shape[0], 1))
	))
	points_rt = np.multiply(points_rt, transf)
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

	grid_titles = {}
	grid_shapes = {}
	axi = {}  # dict of subplots
	plots_height = len(args.camera_list)
	#plots_width = len(args.view_list) + 1
	plots_width = len(args.view_list)
	for ind in range(len(args.camera_list)):
		i = args.camera_list[ind]
		axi.setdefault(i, {})
		for j, v in enumerate(args.view_list):
			m = mode_name[int(v)]
			axi[i].setdefault(m, {})
			grid_index = plots_height*j +ind+1
			grid_title = cam_name[i] + '_' + m
			CaptureSettings = camera_details[i]['CaptureSettings'][0]
			image_shape = CaptureSettings["Height"], CaptureSettings["Width"]

			axi[i][int(v)] = (plots_width, plots_height, grid_index, grid_title, image_shape)
			grid_titles[grid_index] = grid_title
			grid_shapes[grid_index] = image_shape

	grid_display = cv2_grid_display(plots_width, plots_height, grid_titles, grid_shapes, scale_factor=0.5)
	
	for i, row in df.iterrows():
		final_points = np.array([[0, 0, 0], ])
		files = row['ImageFile'].split(";")
		files_path = list(map(lambda x: os.path.join(args.recording_path, 'images', x), files))
		#print(files)
		for j, f in enumerate(files_path):
			cam_id, img_format = files[j].split("_")[2:4]
			img_format = int(img_format)
			if cam_id in args.camera_list and str(img_format) in args.view_list:
				if f.endswith('.ppm'):
					img = Image.open(f)
					img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
				elif f.endswith('.pfm'):
					img, scale = airsim.read_pfm(f)
				else:
					print("Unknown format")

				# print(j, img_format)
				#if cam_id=='0' and img_format==4 and args.plot_3D:
				if img_format==4 and args.plot_3D:
					points_rt = compute_points(img)
					rot_mat = Rotation.from_euler('xyz', angles=[camera_details[cam_id]['Pitch'], camera_details[cam_id]['Roll'], camera_details[cam_id]['Yaw']], degrees=True).as_matrix()
					trans_mat = np.array([camera_details[cam_id]['X'], camera_details[cam_id]['Y'], camera_details[cam_id]['Z']])
					# print('----')
					# print(rot_mat.shape)
					# print(trans_mat.shape)
					# print(points_rt[0].shape)

					translate = lambda p: rot_mat @ p + trans_mat

					points_rt = np.array([translate(p) for p in points_rt])
					final_points = np.concatenate((final_points, points_rt), )

				# if img_format==7 or img_format==5: # Infrared
				# 	img = cv2.blur(img,(5,5))

				
				plots_width, plots_height, grid_index, grid_title, image_shape = axi[cam_id][img_format]
				if len(img.shape)==2:
					img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
				grid_display.imshow(grid_index, img)
		
		if args.plot_3D:
			point_cloud_array.put(final_points)
		
		grid_display.display()
		key = cv2.waitKey(1)
		if key == ord('q'):
			cv2.destroyAllWindows()
			return

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