# Face Mask Detection System 😷

This is a Python-based Face Mask Detection Desktop Application built with `Tkinter` and `YOLOv5`. The application allows users to upload an image or turn on their device's webcam to perform real-time face mask detection. It highlights whether individuals in the frame are wearing masks or not, along with a confidence score.

## Features

- **Upload Image**: Select a local image file and detect face masks instantly.
- **Webcam Detection**: Open your device's camera for real-time video detection.
- **Confidence Scores**: Displays the results showing the detected classes and accuracy percentage.
- **Local Model Loading**: Uses a custom trained YOLOv5 weights file (`mask_yolov5.pt`), which is loaded locally using PyTorch Hub.

## Requirements

You can install all the required libraries quickly using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

*(Alternatively, manually install: `torch torchvision opencv-python pillow pandas requests customtkinter`)*

## Dataset

The model was trained on a Face Mask Detection dataset. To keep this repository lightweight, the dataset images have not been included.

You can download the dataset used for training from this link:
**[https://www.kaggle.com/datasets/andrewmvd/face-mask-detection]**

## How to Run

1. Clone this repository onto your machine.
   *Note: This repository requires the `yolov5-master` folder locally to run. If not included in the repository, you can download it directly from the [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5) and extract the folder matching the path used in the code.*
2. Ensure `mask_yolov5.pt` weights file is located in the root directory.
3. Run the application:

```bash
python mask_detection.py
```

## Technologies Used

- [YOLOv5](https://github.com/ultralytics/yolov5) for Object Detection
- **PyTorch** for model loading and inference
- **OpenCV** for video and frame processing
- **Pillow** for image manipulation
- **Tkinter** for the Graphical User Interface (GUI)
