from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import os

# get vehicle classes: bicycle, car, motorcycle, airplane, bus, train, truck, boat
class_list = [1, 2, 3, 4, 5, 6, 7, 8]
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture('vid2.mp4')
track_history = defaultdict(lambda: [])

save_dir = "images/"  # Directory to save the crops
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
frame_count = 0  # Initialize a frame counter

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, show=False, verbose=False, conf=0.4, classes=class_list, persist=True)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 40:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)


            class_ids = []
            boxes_id = results[0].boxes.id
            if len(boxes_id) > 1:
                id_values = boxes_id.tolist()
            else:
                id_value = boxes_id.item()

            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            for box, cls in zip(boxes_id, clss):
                class_ids.append(cls)
                if cls == 0.0:
                    print("People detect")

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()