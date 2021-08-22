import airsim
import cv2
import numpy as np
import os
import time
import math
import pprint
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera_list', nargs='+', default=['0', '1', '2', '3', '4'], help='List of cameras visualised : [0, 1, ... , 4]')
args = parser.parse_args()

pp = pprint.PrettyPrinter(indent=4)


client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

for camera_name in range(5):
    camera_info = client.simGetCameraInfo(str(camera_name))
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
        #airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
        #airsim.ImageRequest("4", airsim.ImageType.Scene, False, False),
        #airsim.ImageRequest("4", airsim.ImageType.DisparityNormalized),
        #airsim.ImageRequest("4", airsim.ImageType.SurfaceNormals)
        ])

    for i, response in enumerate(responses):
        #print(dir(response))
        #filename = os.path.join(tmp_dir, str(x) + "_" + str(i))
        if response.pixels_as_float:
            #print("pixels_as_float Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
            #airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            img = airsim.get_pfm_array(response)
        else:
            #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            #airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            img = response.image_data_uint8
        img = np.fromstring(img, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        #print(img)
        if response.camera_name in cam_name_show:
            cv2.imshow(cam_name[response.camera_name], img)
            cv2.waitKey(1)

