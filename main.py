"""
@brief: Vehicle reidentification using YOLOv8 and feature-based similarity
@author: Harshit Kumar, Khushi Neema
"""

import cv2
import os
import argparse
from timeit import default_timer
from ultralytics import YOLO
from torchvision import models
from torchreid.utils import FeatureExtractor
from feature_extraction import *
from similarity import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-r", "--reference_img", required=True, help="path to reference image")
parser.add_argument("-f", "--feature", default="hog", choices=["hog", "histogram", "sift", "resnet", "osnet", "composite"],
                    help="Feature type to use for comparison")
args = parser.parse_args()

# Similarity thresholds
hog_threshold = 0.5
histogram_threshold = 0.85
cosine_threshold = 0.1
sift_threshold = 0.2
composite_threshold = 0.5 

# Initialize YOLO model
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(args.input)

# Directory for saving images of matched vehicles
save_dir = "images/"
os.makedirs(save_dir, exist_ok=True)

# Load reference vehicle image and resize
reference_image_path = args.reference_img
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    raise FileNotFoundError(f"Reference image at {reference_image_path} not found.")

if args.feature == "hog" or args.feature == "histogram":
    # Standardize image size for feature extraction
    standard_size = (128, 64)  # Typical size used for HOG feature extraction
    # Resize reference image
    reference_image = cv2.resize(reference_image, standard_size)

start_time = default_timer()
# Compute features for the reference image
if args.feature == "hog":
    reference_hog_features = compute_hog_features(reference_image)
elif args.feature == "histogram":
    reference_color_histogram = compute_color_histogram(reference_image)
elif args.feature =="sift":
    sift = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
elif args.feature == "resnet":
    # Load a pre-trained ResNet model
    model_resnet = models.resnet50(pretrained=True)
    model_resnet.eval()  # Set model to evaluation mode
    reference_dnn_features = extract_dnn_features(model_resnet, reference_image)
elif args.feature == "osnet":
    model_torchreid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='models/osnet_x1_0_imagenet.pth',
        device='cpu'
    )
    reference_dnn_features = extract_torchreid_features(model_torchreid, reference_image)
elif args.feature == "composite":
    # SIFT features
    sift = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
    # OSNet features
    model_torchreid = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='models/osnet_x1_0_imagenet.pth',
        device='cpu'
    )
    reference_dnn_features = extract_torchreid_features(model_torchreid, reference_image)

end_time = default_timer()
reference_feature_time = end_time - start_time

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

            # save all image
            # crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}.jpg"
            # crop_path = os.path.join(save_dir, crop_name)
            # cv2.imwrite(crop_path, crop_img)

            # resize image for non-DNN features
            if args.feature == "hog" or args.feature == "histogram":
                crop_img = cv2.resize(crop_img, standard_size)  # Resize to standard size

            match = False
            current_feature_time = 0.0
            # time for feature extraction
            start_time = default_timer()
            if args.feature == "hog":
                # Compare HOG features
                hog_features = compute_hog_features(crop_img)
                l2_dist = l2_distance(hog_features, reference_hog_features)
                match = l2_dist < hog_threshold
            elif args.feature == "histogram":
                # Compare color histograms
                color_histogram = compute_color_histogram(crop_img)
                hist_dist = histogram_distance(color_histogram, reference_color_histogram)
                match = hist_dist > histogram_threshold
            elif args.feature=="sift":
                #Compare SIFT features
                keypoints, descriptors = sift.detectAndCompute(crop_img, None)
                if descriptors is not None and reference_descriptors is not None:
                    matches = matcher.knnMatch(reference_descriptors, descriptors, k=2)
                    # Apply ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                    # Check if enough good matches were found
                    match = len(good_matches) > sift_threshold * len(reference_descriptors)
            elif args.feature == "resnet":
                # Compare DNN features
                dnn_features = extract_dnn_features(model_resnet, crop_img)
                resnet_cosine_dist = cosine_distance(dnn_features, reference_dnn_features)
                match = resnet_cosine_dist < cosine_threshold
            elif args.feature == "osnet":
                dnn_features = extract_torchreid_features(model_torchreid, crop_img)
                osnet_cosine_dist = cosine_distance(dnn_features, reference_dnn_features)
                match = osnet_cosine_dist < cosine_threshold
            elif args.feature == "composite":
                # reset match
                match = False
                # SIFT features
                keypoints, descriptors = sift.detectAndCompute(crop_img, None)
                if descriptors is not None and reference_descriptors is not None:
                    matches = matcher.knnMatch(reference_descriptors, descriptors, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                # OSNet features
                dnn_features = extract_torchreid_features(model_torchreid, crop_img)
                osnet_cosine_dist = cosine_distance(dnn_features, reference_dnn_features)

                # Composite distance calculation
                comp_dist = composite_distance(good_matches, reference_keypoints, osnet_cosine_dist)
                match = comp_dist < composite_threshold

            end_time = default_timer()
            current_feature_time = end_time - start_time
            total_time = reference_feature_time + current_feature_time
            # print(f"Time taken for {args.feature}: {total_time:.3f}s")

            if match:
                # id
                print(f"Matched vehicle with ID: {track_ids[0]}")
                # Draw a green rectangle around matched vehicles
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Use xyxy format
                cv2.putText(annotated_frame, 'Matched', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

                crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}_{args.feature}.jpg"
                crop_path = os.path.join(save_dir, crop_name)
                cv2.imwrite(crop_path, crop_img)


        cv2.imshow(f"YOLOv8 Tracking with ReID Matching - {args.feature}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()