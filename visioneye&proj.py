from ultralytics import YOLO
import cv2
import numpy as np
import math
import serial.tools.list_ports

# Start webcam
cap = cv2.VideoCapture("https://192.168.0.35:8080/video")
cap.set(3, 640)
cap.set(4, 480)

#visioneye distance

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('visioneye-distance-calculation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center_point = (0, h)
pixel_per_meter = 1000

txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=bbox_clr)
            annotator.visioneye(box, center_point)

            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)    # Bounding box centroid

            distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2))/pixel_per_meter

            text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX,1.2, 3)
            cv2.rectangle(im0, (x1, y1 - text_size[1] - 10),(x1 + text_size[0] + 10, y1), txt_background, -1)
            cv2.putText(im0, f"Distance: {distance:.2f} m",(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color, 3)

    out.write(im0)
    cv2.imshow("visioneye-distance-calculation", im0)

#start ultrasonic


ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portlist = []

for onePort in ports:
    portlist.append(str(onePort))
    print(str(onePort))

val = input("select Port: COM")

for x in range(0,len(portlist)):
    if portlist[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portlist[x])

serialInst.baudrate = 115200
serialInst.port = portVar
serialInst.open()


# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Divide the frame into three equal vertical parts
    frame_height, frame_width, _ = img.shape
    part_width = frame_width // 3

    # Initialize colors for each part
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Coordinates
    for r in results:
        boxes = r.boxes
    
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Calculate horizontal center
            horizontal_center = (x1 + x2) // 2  # Calculate the horizontal center of the bounding box

            # Calculate vertical center
            vertical_center = (y1 + y2) // 2  # Calculate the vertical center of the bounding box

            # Determine which part the bounding box belongs to
            part = horizontal_center // part_width

            # Put box in cam with color based on part
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[part], 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            #receive us
            if serialInst.in_waiting:
              packet = serialInst.readline()
              u = packet.decode('utf')
              print(u)
              num = int(float(u))

            

            # Calculate the base and height for the angle calculation
            base = abs(horizontal_center - x1)
            
            height = distance  # You specified the height as 10

            # Calculate the angle using the tangent function only for green boxes
            if colors[part] == (0, 255, 0):
                angle_rad = math.atan2(height, base)
                angle_deg = math.degrees(angle_rad)
                ans = 90 - angle_deg
                print("Angle --->", ans)

                # Display horizontal center as a green dot
                cv2.circle(img, (horizontal_center, vertical_center), 5, (0, 255, 0), -1)  # Green dot

                # Display angle text above the bounding box
                angle_str = f"Angle: {ans:.2f} degrees"
                org_angle = (x1, y1 - 10)  # Adjust the text position above the bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5  # Adjust the font size
                color = (255, 0, 0)
                thickness = 1
                cv2.putText(img, angle_str, org_angle, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()