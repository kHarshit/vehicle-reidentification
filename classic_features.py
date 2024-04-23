import cv2
import numpy as np
import os
import argparse
from collections import defaultdict
from skimage.feature import hog
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-r", "--reference_img", required=True, help="path to reference image")
parser.add_argument("-f", "--feature", default="hog", choices=["hog", "histogram", "dnn"],
                    help="Feature type to use for comparison (hog or histogram)")
args = parser.parse_args()

# Initialize YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(args.input)

if args.feature == "dnn":
    # Load a pre-trained ResNet model
    model_resnet = models.resnet50(pretrained=True)
    model_resnet.eval()  # Set model to evaluation mode

# Function to preprocess image and extract features
def extract_dnn_features(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model_resnet(image)
    return features.flatten()

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

# Functions to compute features
def compute_hog_features(image):
    """
    Compute HOG features for the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    # normalize
    features /= np.linalg.norm(features)
    return features

def compute_color_histogram(image, bins=32):
    """
    Compute a color histogram for the image
    """
    histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(image.shape[2])]
    histogram = np.concatenate(histogram)
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def cosine_distance(feature1, feature2):
    """
    Compute cosine distance between two features.
    """
    # Reshape features to 2D array for sklearn function
    feature1 = feature1.reshape(1, -1)
    feature2 = feature2.reshape(1, -1)
    return 1 - cosine_similarity(feature1, feature2)[0][0]

# Compute features for the reference image
reference_hog_features = compute_hog_features(reference_image)
reference_color_histogram = compute_color_histogram(reference_image)
if args.feature == "dnn":
    reference_dnn_features = extract_dnn_features(reference_image)

# Similarity thresholds (these values might need tuning based on your specific use case)
hog_threshold = 0.5  # Adjust based on experimentation
histogram_threshold = 0.2

def compare_features(hog1, hog2, hist1, hist2):
    # Calculate distances (smaller values mean more similar)
    hog_distance = np.linalg.norm(hog1 - hog2)
    histogram_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f"HOG Distance: {hog_distance}, Histogram Distance: {histogram_distance}")
    
    # Check against thresholds
    return hog_distance < hog_threshold and histogram_distance > histogram_threshold

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
            if args.feature == "dnn":
                dnn_features = extract_dnn_features(crop_img)

                # Compare features
                # if compare_features(hog_features, reference_hog_features, color_histogram, reference_color_histogram):
                cosine_dist = cosine_distance(dnn_features, reference_dnn_features)
                if cosine_dist < 0.1:
                    # id
                    print(f"Matched vehicle with ID: {track_ids[0]}")
                    print(f"cosine distance: {cosine_dist}")
                    # Draw a green rectangle around matched vehicles
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Use xyxy format
                    cv2.putText(annotated_frame, 'Matched', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()