import numpy as np
import os

###
# Plotter Code
###
#
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys

import math

global POINTS
POINTS = []

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5

global min_height, max_height, colors, calculated, max_dist, min_dist, dist_range
calculated = False


def update_graph():
    global graph_region, POINTS
    global min_height, max_height, colors, calculated, max_dist, min_dist, dist_range
    colors = np.ones(shape=(len(POINTS), 3), dtype=np.uint8)
    #POINTS = [(0,0,1), ]
    for i in range(len(POINTS)):
        #colors[i][0] = 0.
        #colors[i][1] = 0.
        #colors[i][2] = 1.
        pass
        """
        d = dist(POINTS[i], [0,0,0])
        print(d)
        if 30<d<45:
            colors[i][0] = 0.
            colors[i][1] = 0.
            colors[i][2] = 1.
        else:
            colors[i][0] = 0.5
            colors[i][1] = 0.5
            colors[i][2] = 0.5
            # print(POINTS[i])
        """

    if len(POINTS)>0:
        POINTS = np.array(POINTS)
        #graph_region.setData(pos=POINTS, color=colors)
        graph_region.setData(pos=POINTS)


def start_graph():
    print("Setting up graph")
    global app, graph_region, w, g, d3, t
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.resize(800, 600)
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('LIDAR Point Cloud')

    g = gl.GLGridItem()
    w.addItem(g)

    graph_region = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), color=(0, 1, 0, 0.5), size=0.05, pxMode=False)
    #graph_region.rotate(180, 1, 0, 0)
    #graph_region.translate(0, 0, 2.4)
    w.addItem(graph_region)
    t = QtCore.QTimer()
    t.timeout.connect(update_graph)
    t.start(50)

    QtGui.QApplication.instance().exec_()
    print("\n[STOP]\tGraph Window closed. Stopping...")


def lidar_measurement_to_np_array(lidar_measurement):
    data = list()
    for location in lidar_measurement:
        data.append([location.x, location.y, location.z])
    return np.array(data).reshape((-1, 3))

def plot_points(data):
    #try:
    global POINTS
    POINTS = np.array(data)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        start_graph()