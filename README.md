# VisionEye Distance and Object Tracking

This project implements real-time object tracking and distance measurement using YOLOv8 and OpenCV. It processes webcam or IP camera video streams, detects objects, and calculates their distance from a reference point. Additionally, it integrates ultrasonic sensor data via serial communication for enhanced distance estimation.

## Features
- **YOLOv8 Object Detection & Tracking**
- **Distance Calculation** using image processing and geometry
- **Angle Estimation** for object positioning
- **Ultrasonic Sensor Integration** via serial communication
- **Real-time Visualization** with bounding boxes and distance annotations
- **Video Output Recording**

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install ultralytics opencv-python numpy pyserial
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/visioneye-tracking.git
   cd visioneye-tracking
   ```
2. Download the YOLOv8 model weights:
   ```sh
   mkdir yolo-Weights
   wget -O yolo-Weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.0/yolov8n.pt
   ```
3. Ensure your webcam or IP camera is accessible.
4. Run the script:
   ```sh
   python visioneye.py
   ```

## Usage
- The script processes video frames, detects objects, and calculates their distance.
- The bounding box color indicates object position.
- Press `q` to exit the program.

## Configuration
Modify the following parameters in `visioneye.py` as needed:
- **Camera Source**: Change the URL in `cv2.VideoCapture()` for IP cameras.
- **Pixel Per Meter**: Adjust `pixel_per_meter` for accurate distance calculations.
- **Serial Port**: Ensure correct COM port selection for ultrasonic sensors.

## Troubleshooting
- If video feed does not load, check camera URL or webcam connection.
- Ensure `yolov8n.pt` exists in the `yolo-Weights/` directory.
- For serial communication errors, verify the correct port in the script.

## License
This project is licensed under the MIT License.

