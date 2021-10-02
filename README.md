# Camera Setup

Stereo experiment involves 2 cameras. Goal of the experiment is to use monodepth, monodepth2, manydepth and other self-supervised disparity estimation techniques to show that the respective networks can improve on unseen data during runtime as new data comes in.

| Cam | ROI                | FOV | x     | y     | z    | yaw |
|-----|--------------------|-----|-------|-------|------|-----|
| 0   | FrontL             | 90  | 0.25  | -0.16 | -1.7 | 0   |
| 1   | FrontR             | 90  | -1.25 | 0.16  | -1.7 | 0   |


# Getting Started 

## Download Airsim

Download the Airsim 1.4.0 binaries from github : https://github.com/microsoft/AirSim/releases/tag/v1.4.0-linux

The latest 1.5.0 binaries are buggy.


## Setting up Python env

The environment involes installing the headless version of OpenCV as the normal version causes PyQt5 to crash. This script creates a Virtual Env at `~/auto`
```bash
./setup_env.sh
source ~/auto/bin/activate
```

## Launching the Project

The `main.py` will start the sim and the rendering software as multiple processes

```
python main.py
```

## Install PyTorch with CUDA

Refer to the official website for details

https://pytorch.org/get-started/locally/

```bash
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```