import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


def histogram_distance(hist1, hist2):
    """
    Compute histogram distance between two histograms.

    Parameters:
    hist1 (np.ndarray): First histogram
    hist2 (np.ndarray): Second histogram

    Returns:
    float: Histogram distance between the histograms
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def l2_distance(feature1, feature2):
    """
    Compute L2 distance between two features.

    Parameters:
    feature1 (np.ndarray): First feature
    feature2 (np.ndarray): Second feature

    Returns:
    float: L2 distance between the features
    """
    return np.linalg.norm(feature1 - feature2)

def compare_features(hog1, hog2, hist1, hist2, hog_threshold=0.5, histogram_threshold=0.2):
    """
    Compare HOG and color histogram features between two images.
    """
    hog_dist = l2_distance(hog1, hog2)
    histogram_dist = histogram_distance(hist1, hist2)
    print(f"HOG Distance: {hog_dist}, Histogram Distance: {histogram_dist}")
    
    # Check against thresholds
    return hog_dist < hog_threshold and histogram_dist > histogram_threshold

def cosine_distance(feature1, feature2):
    """
    Compute cosine distance between two features.

    Parameters:
    feature1 (np.ndarray): First feature
    feature2 (np.ndarray): Second feature

    Returns:
    float: Cosine distance between the features
    """
    # Reshape features to 2D array for sklearn function
    feature1 = feature1.reshape(1, -1)
    feature2 = feature2.reshape(1, -1)
    return 1 - cosine_similarity(feature1, feature2)[0][0]

