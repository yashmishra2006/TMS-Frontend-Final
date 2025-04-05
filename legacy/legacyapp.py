from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import torch
import json
import time

# Import modules for detection, tracking, and speed estimation
from calibration.calibration_utils import calibrate_camera, undistort_image
from detection.yolo_utils import load_yolo_model, detect_vehicles
from tracking.sort import Sort
from speed_estimation.speed_utils import estimate_speed

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load YOLO Model
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = load_yolo_model("detection/yolov8x.pt", device=device)

# Get class names from YOLO dataset
CLASS_NAMES = yolo_model.names  # Official YOLO class names

# Load camera calibration data
try:
    calib_data = np.load("calibration/calibration_data.npz")
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
except FileNotFoundError:
    raise Exception("Calibration data not found!")

# Compute Perspective Transform
def compute_perspective_transform():
    image_points = np.float32([[800, 410], [1125, 410], [1920, 850], [0, 850]])
    real_points = np.float32([[0, 0], [32, 0], [32, 140], [0, 140]])
    return cv2.getPerspectiveTransform(image_points, real_points)

perspective_matrix = compute_perspective_transform()

# Initialize SORT Tracker
tracker = Sort()

# Path to video file
video_path = "data/videos/traffic_video.mp4"

# Dictionary to store vehicle data
vehicles = {}

# Time threshold to remove old vehicles (in seconds)
VEHICLE_TIMEOUT = 5.0

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "data: " + json.dumps({"error": "Could not open video"}) + "\n\n"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    time_interval = 1.0 / fps
    time_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = undistort_image(frame, camera_matrix, dist_coeffs)
        detections = detect_vehicles(yolo_model, undistorted_frame)

        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections]) if len(detections) > 0 else np.empty((0, 5))
        tracked_objects = tracker.update(dets)

        current_time = time.time()
        time_labels.append(current_time)

        for i, obj in enumerate(tracked_objects):
            x1, y1, x2, y2, track_id = obj
            class_id = int(detections[i][5]) if len(detections) > i else 0
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
            speed = estimate_speed(obj, perspective_matrix, time_interval)
            
            if track_id not in vehicles:
                vehicles[track_id] = {"x": int(x1), "y": int(y1), "speed": [speed], "type": class_name, "last_seen": current_time}
            else:
                vehicles[track_id]["x"] = int(x1)
                vehicles[track_id]["y"] = int(y1)
                vehicles[track_id]["speed"].append(speed)
                vehicles[track_id]["last_seen"] = current_time
            
            if speed > 80:
                vehicles[track_id]["violation"] = True
            else:
                vehicles[track_id]["violation"] = False
        
        # Remove old vehicles
        vehicles_to_remove = [vid for vid, vdata in vehicles.items() if current_time - vdata["last_seen"] > VEHICLE_TIMEOUT]
        for vid in vehicles_to_remove:
            del vehicles[vid]
        
        # Aggregate data
        vehicle_count = len(vehicles)
        category_counts = {cls: sum(1 for v in vehicles.values() if v["type"] == cls) for cls in CLASS_NAMES}
        high_speed_violations = sum(1 for v in vehicles.values() if v.get("violation", False))
        
        # Compute average speed per vehicle
        speed_data = [sum(v["speed"]) / len(v["speed"]) for v in vehicles.values()]
        speed_bins = list(range(0, 120, 10))
        speed_counts = [sum(1 for s in speed_data if lower <= s < upper) for lower, upper in zip(speed_bins[:-1], speed_bins[1:])]

        yield f"data: {json.dumps({'time_labels': time_labels, 'vehicle_count': vehicle_count, 'vehicle_categories': category_counts, 'high_speed_violations': high_speed_violations, 'speed_bins': speed_bins, 'speed_counts': speed_counts})}\n\n"

        time.sleep(time_interval)
    
    cap.release()

@app.route('/api/detect')
def detect():
    return Response(generate_frames(), mimetype="text/event-stream")

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload_file():
    return render_template("upload.html")

@app.route("/processed", methods=["GET"])
def processed_file():
    return render_template("processed.html")

@app.route("/analysis", methods=["GET"])
def analysis():
    return render_template("analysis.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)