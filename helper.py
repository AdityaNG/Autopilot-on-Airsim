import numpy as np
from tqdm import tqdm

import cv2


class cv2_grid_display:

	def __init__(self, plots_width, plots_height, grid_titles, grid_shapes, scale_factor=1.0) -> None:
		self.scale_factor = scale_factor
		self.plots_width = plots_width
		self.plots_height = plots_height
		self.grid_titles = grid_titles
		self.grid_shapes = grid_shapes
		self.total_width = 0
		self.total_height = 0
		self.row_height = {}
		self.col_width = {}
		self.draw_pos_x = {}
		self.draw_pos_y = {}
		
		for i in range(0, plots_height):
			#selected_cams = list(map(str, range(i+1, i+plots_width+1)))
			selected_cams = list(map(int, range(i+1, i+plots_width+1)))
			total_width = 0
			row_height = 0
			for s in selected_cams:
				total_width += grid_shapes[s][1]
				if row_height == 0:
					row_height = grid_shapes[s][0]
				else:
					assert row_height == grid_shapes[s][0]
			self.row_height[i] = row_height


			if self.total_width==0:
				self.total_width = total_width
			else:
				assert self.total_width == total_width

		for i in range(0, plots_width):
			#selected_cams = list(map(str, range(i+1, i+plots_height+1)))
			selected_cams = list(map(int, range(i+1, i+plots_height+1)))
			total_height = 0
			col_width = 0
			for s in selected_cams:
				total_height += grid_shapes[s][0]
				if col_width == 0:
					col_width = grid_shapes[s][1]
				else:
					assert col_width == grid_shapes[s][1]
			self.col_width[i] = col_width

			if self.total_height==0:
				self.total_height = total_height
			else:
				assert self.total_height == total_height

		for i in range(0, plots_height): # row
			for j in range(0, plots_width): # col
				grid_index = self.plots_height * j + i+1
				self.draw_pos_x[grid_index] = 0
				self.draw_pos_y[grid_index] = 0

				for row_i in range(0, i):
					self.draw_pos_x[grid_index] += self.row_height[row_i]

				for col_j in range(0, j):
					self.draw_pos_y[grid_index] += self.col_width[col_j]
		
		self.frame = np.zeros(shape=(self.total_height, self.total_width, 3), dtype=np.uint8)

		

	def imshow(self, index, img):
		img = img.astype(np.uint8)
		if not (img.shape[0]==self.grid_shapes[index][0] and img.shape[1]==self.grid_shapes[index][1]):
			try:
				img = cv2.resize(img, (self.grid_shapes[index][1], self.grid_shapes[index][0]))
			except Exception as ex:
				print(ex)
		# self.frame[
		# 	self.draw_pos_x[index]:self.draw_pos_x[index]+self.grid_shapes[index][0],
		# 	self.draw_pos_y[index]:self.draw_pos_y[index]+self.grid_shapes[index][1]
		# ] = img
		self.frame[
			self.draw_pos_x[index]:self.draw_pos_x[index]+img.shape[0],
			self.draw_pos_y[index]:self.draw_pos_y[index]+img.shape[1]
		] = img
		

	def display(self):
		final_frame = cv2.resize(self.frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)
		cv2.imshow('frame', final_frame)

def euc_dist(p1, p2):
	return np.linalg.norm(p1 - p2)

def compute_point_forces(p: np.array, B: np.array, force_fn: callable):
	"""
	Input:
		A single point p
		A point clouds B
	"""
	assert p.shape == B.shape[1:]
	def force_on_point(p1):
		return force_fn(p1, p)
		
	return np.mean(np.apply_along_axis(force_on_point, axis=1, arr=B), axis=0)

def pc_forces_A2B(A: np.array, B: np.array, force_fn: callable):
	"""
	Input:
		2 point clouds A and B
	"""
	global progress_count
	assert A.shape == B.shape

	with tqdm(total=A.shape[0]**2 //2) as pbar:
		progress_count = 0
		
		def compute_pc_forces(p: np.array):
			global progress_count
			assert p.shape == B.shape[1:]
			progress_count+=1
			pbar.update(progress_count)
			return compute_point_forces(p, B, force_fn)

		return np.apply_along_axis(compute_pc_forces, axis=1, arr=A)


def gravity(p1, p2):
	d = np.linalg.norm(p1 - p2)
	if d==0:
		return np.zeros_like(p1)
	return 1.0/d**2 * (p1 - p2)/d

def rubber_band(p1, p2):
	d = np.linalg.norm(p1 - p2)
	if d==0:
		return np.zeros_like(p1)
	return 1.0/d * (p1 - p2)/d

def stretch_and_contort(A: np.array, B: np.array, A_orig: np.array, B_orig: np.array, theta=10**-3, phi=10**-3, iterations=1):
	"""
	Input:
		2 point clouds A and B
		theta: Inter point cloud attraction
		phi: Intra point cloud attraction
		number of iterations
	"""
	assert A.shape == B.shape
	print(A.shape)
	C = np.zeros_like(A)

	for _ in range(iterations):
		forces_on_A = theta*pc_forces_A2B(A,B,gravity) + phi*pc_forces_A2B(A,A_orig,rubber_band)
		forces_on_B = theta*pc_forces_A2B(B,A,gravity) + phi*pc_forces_A2B(B,B_orig,rubber_band)
		# forces_on_A = theta*pc_forces_A2B(A,B,gravity) + theta*pc_forces_A2B(A,A_orig,gravity)
		# forces_on_B = theta*pc_forces_A2B(B,A,gravity) + theta*pc_forces_A2B(B,B_orig,gravity)
		# forces_on_A = np.mean(forces_on_A, axis=0)
		# forces_on_B = np.mean(forces_on_B, axis=0)
		A = A + forces_on_A
		B = B + forces_on_B

	return A, B

if __name__=="__main__":

	# A = np.array([
	# 	[0,0,0],
	# 	[1,1,1],
	# 	[2,2,2],
	# 	[3,3,3],
	# 	[4,4,4],
	# ])
	# B = np.array([
	# 	[0,1,0],
	# 	[1,2,1],
	# 	[2,3,2],
	# 	[3,4,3],
	# 	[4,5,4],
	# ])
	from math import sin, cos, pi
	from multiprocessing import Process, Queue

	import plotter

	point_cloud_array = Queue()

	def run_example(point_cloud_array):
		A = []
		for i in range(0,360):
			A.append((cos(i*pi/180), sin(i*pi/180), 0.0))
		A = np.array(A)
		B = A.copy()
		B += [0,0.1,0]
		B *= 0.95
		A_orig = A.copy()
		B_orig = B.copy()
		gravitational_force_constant = 10**-3
		rubber_band_force_constant = 0
		point_cloud_array.put(np.concatenate((A, B), ))
		A_new, B_new = stretch_and_contort(A, B, A_orig, B_orig, gravitational_force_constant, rubber_band_force_constant)
		point_cloud_array.put(np.concatenate((A_new, B_new), ))

		for i in range(10):
			A_new, B_new = stretch_and_contort(A_new, B_new, A_orig, B_orig, theta=gravitational_force_constant, phi=rubber_band_force_constant)
			point_cloud_array.put(np.concatenate((A_new, B_new), ))


	run_example_proc = Process(target=run_example, args=(point_cloud_array, ))
	run_example_proc.start()
	
	# Start blocking start_graph call
	plotter.start_graph(point_cloud_array)
