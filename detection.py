import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    # Preprocess the frame for YOLO model
    results = model(frame)

    # Extract detected objects information
    objects = []
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        objects.append((x1, y1, x2, y2, conf, label))
    return objects

def draw_objects(frame, objects):
    for (x1, y1, x2, y2, conf, label) in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def main():
    # Open a connection to the video stream (0 for webcam)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        objects = detect_objects(frame)

        # Draw bounding boxes around detected objects
        frame = draw_objects(frame, objects)

        # Display the frame
        cv2.imshow("Object Detection and Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
