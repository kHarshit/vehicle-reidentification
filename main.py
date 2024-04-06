from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")  # load model

video_path = "vid1.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
frame_count = 0  # Initialize a frame counter
save_dir = "images/"  # Directory to save the crops
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

while ret:
    ret, frame = cap.read()
    frame_count += 1  # Increment the frame counter

    if ret:
        results = model.track(frame, persist=True)
        for i, box in enumerate(results[0].boxes):
            det = box.xyxy
            print(box)
            # remove extra dimension
            det = det[0]
            objectClass = box.cls.item()
            # get vehicle classes: bicycle, car, motorcycle, airplane, bus, train, truck, boat
            if objectClass in [1, 2, 3, 4, 5, 6, 7, 8]:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[y1:y2, x1:x2]  # Crop the detected object

                # Save the cropped image with the desired naming convention
                crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}.jpg"
                crop_path = os.path.join(save_dir, crop_name)
                cv2.imwrite(crop_path, crop)

        frame_ = results[0].plot()

        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()