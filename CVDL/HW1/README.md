# Computer Vision and Deep Learning - Homework 1

This project implements five computer vision tasks as part of a CVDL course: Camera Calibration, Augmented Reality, Stereo Disparity Mapping, SIFT Feature Detection, and CIFAR10 Classification with VGG19-BN. All components are integrated into a single PyQt5 interface for demonstration.

## Requirements

- Python 3.8
- OpenCV-contrib-python (3.4.2.17)
- Matplotlib 3.7.3
- PyQt5 (5.15.10)
- PyTorch 2.1.0
- Torchvision 0.16.0
- Torchsummary 1.5.1
- Tensorboard 2.14.0
- Pillow 10.1.0

Install the required packages using:

```bash
pip install opencv-contrib-python==3.4.2.17
pip install matplotlib==3.7.3
pip install pyqt5==5.15.10
pip install torch==2.1.0 torchvision==0.16.0
pip install torchsummary==1.5.1
pip install tensorboard==2.14.0
pip install pillow==10.1.0
```

## Project Structure

```
HW1/
├── Dataset_CVDL_Hw1/  # Dataset folder for the assignment
├── docs/              # Documentation files
├── src/               # Source code
│   ├── log/           # Log files
│   ├── HW1_controler.py  # Main controller for the application
│   ├── HW1_UI.py      # UI implementation functions
│   ├── HW1.ui         # PyQt5 UI file
│   ├── myVGG19.py     # VGG19 model implementation
│   └── Train_VGG19_BN.ipynb  # Jupyter notebook for model training
└── README.md          # Project documentation
```

## Features

### 1. Camera Calibration

Implementation of camera calibration processes including:
- Corner detection
- Finding the intrinsic matrix
- Finding the extrinsic matrix
- Finding the distortion matrix
- Image undistortion

Utilizing OpenCV functions such as `cv2.findChessboardCorners`, `cv2.cornerSubPix`, `cv2.calibrateCamera`, and `cv2.undistort`.

### 2. Augmented Reality

Displays text on a chessboard:
- Horizontal text display on a chessboard
- Vertical text display on a chessboard

Uses alphabet library files (`alphabet_lib_onboard.txt` and `alphabet_lib_vertical.txt`) and employs `cv2.projectPoints` to calculate the projection of 3D points onto the 2D plane for drawing text.

### 3. Stereo Disparity Map

Utilizes stereo vision techniques to calculate depth information:
- Generating stereo disparity maps
- Checking disparity values

Creates a stereo matching object with `cv2.StereoBM_create` to compute disparity maps between left and right images, and implements functionality to display corresponding points on the right image when clicking on the left image.

### 4. SIFT Feature Detection

Uses the SIFT algorithm for feature detection and matching:
- Keypoint detection
- Keypoint matching

Creates SIFT detector with `cv2.SIFT_create()`, performs feature matching with `cv2.BFMatcher`, and visualizes the results using `cv2.drawKeypoints` and `cv2.drawMatchesKnn`.

### 5. CIFAR10 Classifier with VGG19

Trains a CIFAR10 classifier using VGG19 architecture with batch normalization:
- Data augmentation and visualization
- Model structure display
- Training/validation accuracy and loss visualization
- Inference using the trained model

Implements the VGG19 model with `torchvision.models.vgg19_bn` and trains it on the CIFAR10 dataset using PyTorch.

## Usage

1. Run the main program:
    ```bash
    python HW1_controler.py
    ```

2. The main interface includes five feature areas, click the corresponding buttons to perform different functions:
   - Camera calibration area: Load images or folders, perform corner detection, calculate parameters, etc.
   - Augmented reality area: Load chessboard images, input text, and display on the chessboard
   - Stereo disparity map area: Load left and right images, calculate disparity maps, and check disparity values
   - SIFT area: Load two images, detect feature points, and perform matching
   - CIFAR10 classifier area: Display augmented images, model structure, accuracy/loss graphs, and perform image inference