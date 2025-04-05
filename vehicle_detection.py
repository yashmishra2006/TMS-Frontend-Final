import os
import cv2
import numpy as np
import torch
import requests
import time
from typing import Any
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.sort import Sort
from speed_estimation.speed_utils import estimate_speed

VALID_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
MAX_SPEED = 250
GRAPH_API_URL = 'http://127.0.0.1:5000/api/update_data'

def send_graph_data_to_api(data: dict) -> None:
    try:
        response = requests.post(GRAPH_API_URL, json=data)
        if response.status_code != 200:
            print(f"Error sending graph data: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Graph API error: {e}")

def compute_perspective_transform() -> np.ndarray:
    image_points = np.float32([[800, 410], [1125, 410], [1920, 850], [0, 850]])
    real_points = np.float32([[0, 0], [32, 0], [32, 140], [0, 140]])
    return cv2.getPerspectiveTransform(image_points, real_points)

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-5)
    return iou

def match_detections_to_tracks(tracked_objects, detections):
    track_to_class = {}
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        best_iou = 0
        best_class = None
        for det in detections:
            dx1, dy1, dx2, dy2, conf, class_id = det
            iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_class = int(class_id)
        if best_class is not None:
            track_to_class[int(track_id)] = best_class
    return track_to_class

def main() -> None:
    try:
        calib_data = np.load("calibration/calibration_data.npz")
        camera_matrix, dist_coeffs = calib_data["camera_matrix"], calib_data["dist_coeffs"]
        print("Loaded calibration data.")
    except FileNotFoundError:
        print("Calibration data not found. Running calibration...")
        _, camera_matrix, dist_coeffs, _, _ = calibrate_camera("calibration/calibration_images", display=False)
        np.savez("calibration/calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("Calibration complete.")

    print("Loading YOLO model...")
    yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)
    class_names = yolo_model.names
    print("YOLO loaded")

    perspective_matrix = compute_perspective_transform()
    tracker = Sort()

    cap: Any = cv2.VideoCapture("data/videos/traffic_video.mp4")
    if not cap.isOpened():
        print("Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25.0 if fps <= 0 or fps > 1000 else fps
    time_interval = 1.0 / fps
    seen_ids = set()
    last_report_time = time.time()

    type_counter = defaultdict(int)
    total_speed = 0
    speed_count = 0
    overspeed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)
        detections = detect_vehicles(yolo_model, undistorted_frame)
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        timestamp = time.strftime('%H:%M:%S', time.gmtime())
        track_to_class = match_detections_to_tracks(tracked_objects, detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            track_id = int(track_id)
            class_id = track_to_class.get(track_id)
            if class_id is None:
                continue
            vehicle_type = class_names[class_id]
            if vehicle_type not in VALID_VEHICLE_CLASSES:
                continue

            speed = estimate_speed(obj, perspective_matrix, time_interval)
            speed = min(speed, MAX_SPEED)

            if track_id not in seen_ids:
                seen_ids.add(track_id)
                type_counter[vehicle_type] += 1
                total_speed += speed
                speed_count += 1
                if speed > 80:
                    overspeed_count += 1

        if time.time() - last_report_time >= 10:
            avg_speed = total_speed / speed_count if speed_count else 0
            graph_data = {
                "vehicle_count": len(seen_ids),
                "vehicle_types": dict(type_counter),
                "average_speed": round(avg_speed, 2),
                "overspeeding": overspeed_count,
                "timestamp": timestamp
            }
            print("Sending graph summary:", graph_data)
            send_graph_data_to_api(graph_data)

            type_counter.clear()
            total_speed = 0
            speed_count = 0
            overspeed_count = 0
            seen_ids.clear()
            last_report_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
