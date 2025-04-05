import numpy as np
import cv2

# ================= HOMOGRAPHY UTILS =================
def compute_homography() -> np.ndarray:
    """
    Computes a homography matrix that maps image coordinates to real-world coordinates.
    
    This function uses hard-coded corresponding points for demonstration purposes.
    Replace these points with your own calibration points or load them from a file if needed.
    
    Returns:
        homography_matrix (np.ndarray): A 3x3 matrix mapping image coordinates to world coordinates.
        
    Raises:
        ValueError: If the homography matrix computation fails.
    """
    # Define 4 corresponding points in the image (pixel coordinates)
    image_points = np.float32([
        [802, 410],   # Top-left
        [1122, 412],  # Top-right
        [1896, 836],  # Bottom-right
        [26, 834]     # Bottom-left
    ])
    
    # Define corresponding points in the real world (e.g., in meters)
    world_points = np.float32([
        [0,   0],   # Top-left  (real-world point)
        [32,  0],   # Top-right
        [32, 140],  # Bottom-right
        [0,  140]   # Bottom-left in world coordinates
    ])
    
    # Compute the homography matrix using OpenCV's findHomography function.
    homography_matrix, status = cv2.findHomography(image_points, world_points)
    
    if homography_matrix is None:
        raise ValueError("Homography matrix computation failed. Check your input points.")
    
    return homography_matrix