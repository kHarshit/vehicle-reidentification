"""
@brief: Feature extraction functions for vehicle reidentification
@author: Harshit Kumar, Khushi Neema
"""

import cv2
import numpy as np
from skimage.feature import hog
import torch
from torchvision import transforms

def compute_hog_features(image):
    """
    Compute HOG features for the image

    Parameters:
    image (np.ndarray): Input image

    Returns:
    np.ndarray: HOG features
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

    Parameters:
    image (np.ndarray): Input image
    bins (int): Number of bins for the histogram

    Returns:
    np.ndarray: Color histogram
    """
    histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(image.shape[2])]
    histogram = np.concatenate(histogram)
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def extract_dnn_features(model, image):
    """
    Extract deep features using a pre-trained model

    Parameters:
    model (torch.nn.Module): Pre-trained model
    image (np.ndarray): Input image

    Returns:
    np.ndarray: Extracted features
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.flatten()
