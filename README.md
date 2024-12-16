# Dolphin Vision System

A machine vision system for identifying and tracking dolphins using YOLOv7 and real-time object tracking.

## Features

- Real-time dolphin detection and tracking
- Separate body and face detection models
- 30 FPS video processing
- Recording and screenshot capabilities
- Interactive debug interface

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Munkres algorithm
- SORT (Simple Online Realtime Tracking)

## Project Structure

```
.
├── cfg/                # Model configurations
├── data/              # Training data
├── models/            # Model architecture
├── utils/             # Utility functions
├── system.py          # Main system implementation
├── train.py           # Training pipeline
├── train_aux.py       # Auxiliary training functions
└── sort.py            # Tracking implementation
```

## Usage

1. Run the main system:
```bash
python system.py --video [VIDEO_PATH] --debug
```

2. Train the model:
```bash
python train.py --data data/dolphins.yaml --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt
```

## Controls

- `Space`: Pause/Resume
- `D`: Toggle debug mode
- `F`: Toggle fullscreen
- `B`: Toggle binary mode
- `S`: Capture screenshot
- `R`: Toggle recording
- `Q`: Quit

## Model Files

The system uses several specialized models:
- `bestbody.pt`: Main body detection model
- `bestface.pt`: Face detection model
- Additional models for different detection scenarios

Note: Model files are not included in the repository due to size constraints.
