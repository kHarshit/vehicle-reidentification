import cv2
import numpy as np
import os
from collections import defaultdict
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture('vid3.mp4')

# Directory for saving images
save_dir = "images/"
os.makedirs(save_dir, exist_ok=True)

# Load reference vehicle image and resize
# reference_image_path = 'frame_7_ID_2.0.jpg'
reference_image_path='reference_vehicle.jpg'
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    raise FileNotFoundError(f"Reference image at {reference_image_path} not found.")

# Standardize image size for feature extraction
standard_size = (128, 64)  # Typical size used for feature extraction

# Resize reference image
reference_image = cv2.resize(reference_image, standard_size)

# Function to compute SIFT features
def compute_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Compute features for the reference image
reference_keypoints, reference_descriptors = compute_sift_features(reference_image)

# Similarity thresholds (these values might need tuning based on your specific use case)
sift_threshold = 0.4  # Adjust based on experimentation

# def compare_features(descriptors1, descriptors2):
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)

#     # Matcher
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     # Ratio test as per Lowe's paper
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good_matches.append(m)

#     # Check if enough good matches are found
#     return len(good_matches) > sift_threshold * len(reference_descriptors)

def compare_features(descriptors1, descriptors2):
    # Check if either descriptor set is empty
    if descriptors1 is None or descriptors2 is None:
        return False

    # Check if both descriptor sets have descriptors
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return False

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Convert descriptors to float32 for FLANN
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Perform FLANN matching
    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error as e:
        print("FLANN error:", e)
        return False

    # Ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Check if enough good matches are found
    return len(good_matches) > sift_threshold * len(reference_descriptors)



# Tracking and matching
frame_count = 0  # Initialize a frame counter
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1  # Increment the frame counter
    results = model.track(frame, show=False, verbose=False, conf=0.4, classes=[2], persist=True)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()  # Use xyxy format
        annotated_frame = results[0].plot()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Use xyxy format
            crop_img = frame[y1:y2, x1:x2]  # Use xyxy format
            # save image
            crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}.jpg"
            crop_path = os.path.join(save_dir, crop_name)
            cv2.imwrite(crop_path, crop_img)
            crop_img = cv2.resize(crop_img, standard_size)  # Resize to standard size

            # Compute SIFT features for the detected vehicle
            keypoints, descriptors = compute_sift_features(crop_img)

            # Compare features
            if compare_features(reference_descriptors, descriptors):
                # id
                print(f"Matched vehicle with ID: {results[0].boxes.id[i]}")
                # Draw a green rectangle around matched vehicles
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Use xyxy format
                cv2.putText(annotated_frame, 'Matched', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
