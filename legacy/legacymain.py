import os
import cv2
import numpy as np
import torch
from typing import Any

# Workaround for OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch uses GPU if available
torch.set_num_threads(1)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Import all required modules
from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.sort import Sort
from speed_estimation.speed_utils import estimate_speed

def compute_perspective_transform() -> np.ndarray:
    """ Computes a perspective transform matrix mapping image coordinates to real-world coordinates. """
    image_points = np.float32([[800, 410], [1125, 410], [1920, 850], [0, 850]])
    real_points = np.float32([[0, 0], [32, 0], [32, 140], [0, 140]])
    return cv2.getPerspectiveTransform(image_points, real_points)

def main() -> None:
    # --- Load Camera Calibration ---
    try:
        calib_data = np.load("calibration/calibration_data.npz")
        camera_matrix, dist_coeffs = calib_data["camera_matrix"], calib_data["dist_coeffs"]
        print("Loaded calibration data.")
    except FileNotFoundError:
        print("Calibration data not found. Running calibration...")
        _, camera_matrix, dist_coeffs, _, _ = calibrate_camera("calibration/calibration_images", display=False)
        np.savez("calibration/calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("Calibration completed and data saved.")

    # --- Load YOLO Model ---
    print("Loading YOLO model...")
    yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)
    print("YOLO model loaded successfully.")

    # --- Get YOLO Class Names ---
    class_names = yolo_model.names  # Dictionary of {id: "class_name"}
    print("Model can detect:", class_names)

    # --- Compute Perspective Transform ---
    perspective_matrix = compute_perspective_transform()
    print("Perspective Transform Matrix:\n", perspective_matrix)

    # --- Initialize SORT Tracker ---
    tracker = Sort()

    # --- Open Video Capture ---
    video_path = "data/videos/traffic_video.mp4"
    cap: Any = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get FPS (fallback to 25 if FPS is invalid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25.0 if fps <= 0 or fps > 1000 else fps
    print("Video FPS:", fps)
    time_interval = 1.0 / fps

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame
        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)

        # Detect vehicles using YOLO
        detections = detect_vehicles(yolo_model, undistorted_frame)

        # Convert detections to SORT format [x1, y1, x2, y2, score]
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if len(detections) > 0 else np.empty((0, 5))

        # Track detected vehicles
        tracked_objects = tracker.update(dets)

        # Store vehicle data
        vehicle_data = []

        # Process each tracked object
        for obj, det in zip(tracked_objects, detections):
            x1, y1, x2, y2, track_id = obj
            class_id = int(det[5])  # Extract class ID
            vehicle_type = class_names.get(class_id, "unknown")  # Get class name

            # Estimate speed
            speed = estimate_speed(obj, perspective_matrix, time_interval)

            # Append to output list
            vehicle_data.append({
                "id": int(track_id),
                "speed": round(speed, 1),
                "type": vehicle_type
            })

        # Print vehicle data in JSON format
        print(vehicle_data)

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
main()