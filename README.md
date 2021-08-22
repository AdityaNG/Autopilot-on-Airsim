# Camera Setup

![Camera Demo](/gifs/cams.gif)

Current setup involves 5 cameras. Four of them being 120 deg wide angle for short range sensing and one narrow FOV front facing camera for long range sensing.

| Cam | ROI                | FOV | x     | y     | z    | yaw |
|-----|--------------------|-----|-------|-------|------|-----|
| 0   | Front, Short range | 120 | 0.25  | 0     | -1.7 | 0   |
| 1   | Back               | 120 | -1.25 | 0     | -1.7 | 180 |
| 2   | Right              | 120 | -0.8  | 0.45  | -1.7 | 90  |
| 3   | Left               | 120 | -0.8  | -0.45 | -1.7 | -90 |
| 4   | Front, Long Range  | 45  | 0.25  | 0     | -1.7 | 0   |


# Getting Started 

## Installing Airsim 1.4.0

Download the Airsim 1.4.0 binaries from github : https://github.com/microsoft/AirSim/releases/tag/v1.4.0-linux


The python client can be installed from pip :

```bash
pip install airsim
```


## Launching Airsim

First cd into the downloaded folder. Then execute while pointing to the full path of the settings.json within this project

```
cd ~/Apps/AirSimNH_1.4.0/LinuxNoEditor
./AirSimNH.sh -WINDOWED -ResX=640 -ResY=480 --settings /home/aditya/Autopilot/settings.json
```
