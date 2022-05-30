import numpy as np

Y_POS_MIN = -0.1
Y_POS_MAX = 0.1
NUM_CAMS = 4
DEFAULT_IMG_WIDTH = 480
DEFAULT_IMG_HEIGHT = 270

INITIAL_TEMPLATE = """
{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "ViewMode": "NoDisplay",
  "Vehicles": {
    "PhysXCar": {
      "VehicleType": "PhysXCar",
      "DefaultVehicleState": "",
      "AutoCreate": true,
      "PawnPath": "",
      "EnableCollisionPassthrogh": false,
      "EnableCollisions": true,
      "RC": {
        "RemoteControlID": -1
      },
      "Cameras": {   """

CAM_TEMPLATE = """
		"{CAM_ID}": {{
			"CaptureSettings": [
				{{
					"ImageType": 0,
					"Width": {IMG_WIDTH},
				  	"Height": {IMG_HEIGHT},
					"FOV_Degrees": 90
				}},
				{{
					"ImageType": 4,
					"Width": {IMG_WIDTH},
				  	"Height": {IMG_HEIGHT},
					"FOV_Degrees": 90
				}}
				],
				"X": 0.25,
				"Y": {Y_POS:.4f},
				"Z": -1.7,
				"Pitch": 0.0,
				"Roll": 0.0,
				"Yaw": 0
        }}"""

FINAL_TEMPLATE = """
		}
    }
  },
  "SubWindows": [
	{"WindowID": 0, "ImageType": 0, "CameraName": "0", "Visible": true},
    {"WindowID": 1, "ImageType": 7, "CameraName": "0", "Visible": true}
  ],
  "Recording": {
    "RecordOnMove": false,
    "RecordInterval": 0.05,
    "Cameras": [
		{ "CameraName": "0", "ImageType": 0, "PixelsAsFloat": false, "Compress": false },
		{ "CameraName": "0", "ImageType": 1, "PixelsAsFloat": false, "Compress": false },
		{ "CameraName": "0", "ImageType": 7, "PixelsAsFloat": false, "Compress": false }
    ]
  }
}"""

def generate():
	res = ""
	res += INITIAL_TEMPLATE
	#print(INITIAL_TEMPLATE)
	for i, y in enumerate(np.arange(Y_POS_MIN, Y_POS_MAX, (Y_POS_MAX-Y_POS_MIN)/ NUM_CAMS)):
		cam = CAM_TEMPLATE.format(CAM_ID=i, Y_POS=y, IMG_WIDTH=DEFAULT_IMG_WIDTH, IMG_HEIGHT=DEFAULT_IMG_HEIGHT)
		#print(cam, end=', \n')
		res += cam + ',\n'

	cam = CAM_TEMPLATE.format(CAM_ID=NUM_CAMS, Y_POS=Y_POS_MAX, IMG_WIDTH=DEFAULT_IMG_WIDTH, IMG_HEIGHT=DEFAULT_IMG_HEIGHT)
	#print(cam, end='\n')
	res += cam + '\n'
	#print(FINAL_TEMPLATE)
	res += FINAL_TEMPLATE
	return res

if __name__=="__main__":
	print(generate())