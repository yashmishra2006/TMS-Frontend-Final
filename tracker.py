from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Optional
import numpy as np
import cv2

class SORTTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3) -> None:
        """
        Initialize the DeepSORT tracker.
        
        Parameters:
          - max_age: Maximum number of frames to keep a track alive without detections.
          - n_init: Number of consecutive detections before confirming a track.
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
    
    def update(self, detections: List[List[float]], frame: Optional[np.ndarray] = None) -> List[List[float]]:
        """
        Update the tracker with the current detections.
        
        Parameters:
          - detections: A list of detections, each in the format [x1, y1, x2, y2, confidence].
          - frame: Optional; the current frame as a NumPy array (useful for appearance extraction).
          
        Returns:
          - A list of tracked objects, each in the format [x1, y1, x2, y2, track_id].
        """
        # Convert detections to the format expected by DeepSORT:
        # Each detection should be: [ [x1, y1, width, height], confidence, class ]
        # We pass None for the class label since it's not available.
        converted_detections = []
        for det in detections:
            if len(det) != 5:
                continue  # Skip invalid detections.
            x1, y1, x2, y2, conf = det
            width = x2 - x1
            height = y2 - y1
            converted_detections.append([[x1, y1, width, height], float(conf), None])
        
        # Update the tracker; pass the current frame if available.
        tracks = self.tracker.update_tracks(converted_detections, frame=frame)
        
        # Process the tracks and filter out unconfirmed tracks.
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            # Retrieve bounding box in [x1, y1, x2, y2] format.
            bbox = track.to_ltrb()  # left, top, right, bottom
            track_id = track.track_id
            tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        
        return tracked_objects
