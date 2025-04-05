import numpy as np
import cv2
import glob
import os

def calibrate_camera(calib_images_path, pattern="calibration*.jpg", checkerboard_size=(8,6), display=False):
    """
    Calibrates the camera using a set of calibration images from the Udacity CarND dataset.
    
    Parameters:
      calib_images_path: Directory containing calibration images.
      pattern: Glob pattern to select calibration images.
      checkerboard_size: Number of inner corners per chessboard row and column (width, height).
      display: If True, displays each image with detected corners.
      
    Returns:
      ret: Calibration success flag.
      camera_matrix: Intrinsic camera matrix.
      dist_coeffs: Distortion coefficients.
      rvecs: Rotation vectors.
      tvecs: Translation vectors.
    """
    # Termination criteria for refining corner detection.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, e.g., (0,0,0), (1,0,0), ...,(7,5,0) for an 8x6 grid.
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    # Create a grid of (x, y) coordinates
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store 3D points in real world space and 2D points in image plane.
    objpoints = []
    imgpoints = []
    
    # Build the full path pattern for calibration images.
    images = glob.glob(os.path.join(calib_images_path, pattern))
    print(f"Found {len(images)} calibration images using pattern '{pattern}'.")
    
    # Use additional flags to improve corner detection in varied lighting conditions.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    for idx, fname in enumerate(images):
        print(f"Processing {fname}...")
        img = cv2.imread(fname)
        if img is None:
            print(f"  -> Could not read image: {fname}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        
        if ret:
            print(f"  -> Corners found in {fname}")
            objpoints.append(objp)
            # Refine corner detection for sub-pixel accuracy.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            if display:
                cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)
        else:
            print(f"  -> Corners NOT found in {fname}")
    
    if display:
        cv2.destroyAllWindows()
    
    if not objpoints or not imgpoints:
        raise ValueError("No valid calibration images were found. Check your images and settings.")
    
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Returns the undistorted version of the input image.
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

if __name__ == "__main__":
    # Path to the folder containing calibration images downloaded from Udacity repo.
    calib_folder = "calibration_images"  # Adjust this path if needed.
    
    # Use the glob pattern matching the Udacity dataset images.
    pattern = "calibration*.jpg"
    
    # The Udacity dataset uses a checkerboard with 8 inner corners across and 6 down.
    checkerboard_size = (8, 6)
    
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            calib_folder, pattern=pattern, checkerboard_size=checkerboard_size, display=True)
        print("Calibration successful.")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
    except ValueError as e:
        print("Calibration failed:", e)
