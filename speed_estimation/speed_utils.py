import numpy as np
import cv2

# Global dictionaries to store previous world positions and smoothed speeds by track_id.
previous_positions = {}
smoothed_speeds = {}

# Smoothing factor: adjust ALPHA between 0 (very smooth) and 1 (no smoothing)
ALPHA = 0.3

def estimate_speed(tracked_obj, homography_matrix, time_interval=0.1):
    """
    Estimate the speed of a tracked object based on its displacement between frames,
    applying exponential moving average filtering to smooth the output.
    
    Parameters:
        tracked_obj (list): Bounding box and ID in the format [x1, y1, x2, y2, track_id].
        homography_matrix (np.ndarray): A 3x3 matrix mapping image coordinates to real-world coordinates.
        time_interval (float): Time elapsed between frames (in seconds).
    
    Returns:
        float: Smoothed estimated speed in km/h.
    """
    # Unpack the bounding box and track ID.
    x1, y1, x2, y2, track_id = tracked_obj

    # Compute the center of the bounding box in pixel coordinates.
    center_pixel = np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1], dtype=np.float32)
    
    # Transform the pixel coordinates to world coordinates using the homography matrix.
    world_point = homography_matrix @ center_pixel
    if world_point[2] == 0:
        # Avoid division by zero if the mapping is invalid.
        return 0.0
    world_point /= world_point[2]  # Normalize from homogeneous coordinates.
    current_world_coords = world_point[:2]
    
    # Compute displacement if a previous position exists.
    if track_id in previous_positions:
        prev_world_coords = previous_positions[track_id]
        displacement = np.linalg.norm(current_world_coords - prev_world_coords)
        speed_mps = displacement / time_interval
        speed_kmph = speed_mps * 3.6  # Convert m/s to km/h.
    else:
        speed_kmph = 0.0

    # Update the previous world position.
    previous_positions[track_id] = current_world_coords

    # Apply exponential moving average (EMA) for smoothing the speed estimate.
    if track_id in smoothed_speeds:
        smoothed_speed = ALPHA * speed_kmph + (1 - ALPHA) * smoothed_speeds[track_id]
    else:
        smoothed_speed = speed_kmph
    smoothed_speeds[track_id] = smoothed_speed
    
    return smoothed_speed

# Optional: For testing purposes only.
if __name__ == "__main__":
    # Dummy tracked object [x1, y1, x2, y2, track_id]
    dummy_tracked_obj = [100, 200, 200, 300, 1]
    
    # Example homography matrix computed from hard-coded points.
    image_points = np.float32([[100, 200], [400, 200], [100, 500], [400, 500]])
    world_points = np.float32([[0, 0], [10, 0], [0, 20], [10, 20]])
    homography_matrix, _ = cv2.findHomography(image_points, world_points)
    
    # Simulate multiple calls to the speed estimator (to see smoothing in action)
    print("Estimated speed (frame 1):", 
          estimate_speed(dummy_tracked_obj, homography_matrix, time_interval=0.1), "km/h")
    
    # Simulate movement: modify the dummy object slightly
    dummy_tracked_obj = [105, 205, 205, 305, 1]
    print("Estimated speed (frame 2):", 
          estimate_speed(dummy_tracked_obj, homography_matrix, time_interval=0.1), "km/h")
    
    # Further simulate movement
    dummy_tracked_obj = [110, 210, 210, 310, 1]
    print("Estimated speed (frame 3):", 
          estimate_speed(dummy_tracked_obj, homography_matrix, time_interval=0.1), "km/h")
