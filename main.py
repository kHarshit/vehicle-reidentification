import cv2
import os
import argparse
from ultralytics import YOLO
from torchvision import models
from feature_extraction import *
from similarity import *

"""
Vehicle Reid

# MEHTODS  
1. HOG (Histogram of Oriented Gradients)
2. Histogram (Color Histogram)
3. DNN (Deep Neural Network)

"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-r", "--reference_img", required=True, help="path to reference image")
parser.add_argument("-f", "--feature", default="hog", choices=["hog", "histogram", "dnn"],
                    help="Feature type to use for comparison")
args = parser.parse_args()

# Initialize YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(args.input)

if args.feature == "dnn":
    # Load a pre-trained ResNet model
    model_resnet = models.resnet50(pretrained=True)
    model_resnet.eval()  # Set model to evaluation mode

# Directory for saving images
save_dir = "images/"
os.makedirs(save_dir, exist_ok=True)

# Load reference vehicle image and resize
reference_image_path = args.reference_img
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    raise FileNotFoundError(f"Reference image at {reference_image_path} not found.")

# Standardize image size for feature extraction
standard_size = (128, 64)  # Typical size used for HOG feature extraction

# Resize reference image
reference_image = cv2.resize(reference_image, standard_size)

# Compute features for the reference image
reference_hog_features = compute_hog_features(reference_image)
reference_color_histogram = compute_color_histogram(reference_image)
if args.feature == "dnn":
    reference_dnn_features = extract_dnn_features(model_resnet, reference_image)

# Similarity thresholds (these values might need tuning based on your specific use case)
hog_threshold = 0.5  # Adjust based on experimentation
histogram_threshold = 0.2

# get vehicle classes: bicycle, car, motorcycle, airplane, bus, train, truck, boat
# class_list = [1, 2, 3, 4, 5, 6, 7, 8]
class_list = [2]
# Tracking and matching
frame_count = 0  # Initialize a frame counter
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1  # Increment the frame counter
    results = model.track(frame, show=False, verbose=False, conf=0.4, classes=class_list, persist=True)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()  # Use xyxy format
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        annotated_frame = results[0].plot()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Use xyxy format
            crop_img = frame[y1:y2, x1:x2]  # Use xyxy format
            # save image
            crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}.jpg"
            crop_path = os.path.join(save_dir, crop_name)
            cv2.imwrite(crop_path, crop_img)
            crop_img = cv2.resize(crop_img, standard_size)  # Resize to standard size

            # Compute features for the detected vehicle
            hog_features = compute_hog_features(crop_img)
            color_histogram = compute_color_histogram(crop_img)
            match = False
            if args.feature == "dnn":
                dnn_features = extract_dnn_features(model_resnet, crop_img)
                cosine_dist = cosine_distance(dnn_features, reference_dnn_features)
                if cosine_dist < 0.1:
                    match = True
            else:
                # Compare features
                match =  compare_features(hog_features, reference_hog_features, color_histogram, reference_color_histogram)
            
            if match:
                # id
                print(f"Matched vehicle with ID: {track_ids[0]}")
                # Draw a green rectangle around matched vehicles
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Use xyxy format
                cv2.putText(annotated_frame, 'Matched', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()