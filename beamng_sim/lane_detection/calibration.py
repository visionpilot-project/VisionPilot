"""
Camera Calibration Module
Following CarND's chessboard-based camera calibration approach.

This module provides functions to:
1. Calibrate camera using chessboard images
2. Save/load calibration parameters
3. Undistort images using calibration
"""

import numpy as np
import cv2
import pickle
import glob
import os


def calibrate_camera(images, chessboard_size=(9, 6), save_file=None):
    """
    Calibrate camera using chessboard images.
    Following CarND's implementation approach.
    
    Args:
        images (list): List of file paths to chessboard calibration images
        chessboard_size (tuple): Number of inner corners (width, height)
        save_file (str): Optional path to save calibration parameters
    
    Returns:
        tuple: (ret, mtx, dist, rvecs, tvecs)
            - ret: RMS re-projection error
            - mtx: Camera matrix
            - dist: Distortion coefficients
            - rvecs: Rotation vectors
            - tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Image shape for calibration
    img_shape = None
    
    successful_images = 0
    print(f"Processing {len(images)} calibration images...")
    
    for idx, fname in enumerate(images):
        # Read image
        img = cv2.imread(fname)
        
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Store image shape
        if img_shape is None:
            img_shape = gray.shape[::-1]
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_images += 1
            print(f"  [{idx+1}/{len(images)}] ✓ Corners found in {os.path.basename(fname)}")
        else:
            print(f"  [{idx+1}/{len(images)}] ✗ No corners in {os.path.basename(fname)}")
    
    print(f"\nSuccessfully processed {successful_images}/{len(images)} images")
    
    if successful_images == 0:
        raise ValueError("No chessboard corners found in any images. Check chessboard_size parameter.")
    
    # Calibrate camera
    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    print(f"Camera calibration complete!")
    print(f"RMS re-projection error: {ret:.4f}")
    
    # Save calibration parameters if requested
    if save_file is not None:
        calibration_data = {
            'mtx': mtx,
            'dist': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'img_shape': img_shape,
            'rms_error': ret
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        with open(save_file, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"Calibration parameters saved to: {save_file}")
    
    return ret, mtx, dist, rvecs, tvecs


def load_calibration(calibration_file):
    """
    Load camera calibration parameters from file.
    
    Args:
        calibration_file (str): Path to pickled calibration file
    
    Returns:
        dict: Calibration data containing mtx, dist, rvecs, tvecs, img_shape, rms_error
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)
    
    print(f"Loaded calibration from: {calibration_file}")
    print(f"RMS error: {calibration_data['rms_error']:.4f}")
    
    return calibration_data


def undistort_image(img, mtx, dist):
    """
    Undistort an image using camera calibration parameters.
    
    Args:
        img (numpy array): Distorted image
        mtx (numpy array): Camera matrix
        dist (numpy array): Distortion coefficients
    
    Returns:
        numpy array: Undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def calibrate_from_folder(folder_path, pattern='*.jpg', chessboard_size=(9, 6), save_file=None):
    """
    Convenience function to calibrate camera from a folder of chessboard images.
    
    Args:
        folder_path (str): Path to folder containing chessboard images
        pattern (str): File pattern to match (default: '*.jpg')
        chessboard_size (tuple): Number of inner corners (width, height)
        save_file (str): Optional path to save calibration parameters
    
    Returns:
        tuple: (ret, mtx, dist, rvecs, tvecs)
    """
    # Find all calibration images
    search_path = os.path.join(folder_path, pattern)
    images = glob.glob(search_path)
    
    if len(images) == 0:
        raise ValueError(f"No images found matching pattern: {search_path}")
    
    print(f"Found {len(images)} calibration images in {folder_path}")
    
    return calibrate_camera(images, chessboard_size, save_file)


# Example usage
if __name__ == "__main__":
    # Example: Calibrate camera from CarND calibration images
    calibration_folder = "../CarND-Advanced-Lane-Lines/camera_cal"
    output_file = "../models/camera_calibration.pkl"
    
    try:
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = calibrate_from_folder(
            calibration_folder,
            pattern='calibration*.jpg',
            chessboard_size=(9, 6),
            save_file=output_file
        )
        
        print("\nCalibration matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist)
        
        # Test undistortion on a sample image
        test_image_path = "../CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg"
        if os.path.exists(test_image_path):
            img = cv2.imread(test_image_path)
            undistorted = undistort_image(img, mtx, dist)
            
            # Save comparison
            comparison = np.hstack([img, undistorted])
            cv2.imwrite("calibration_comparison.jpg", comparison)
            print(f"\nTest undistortion saved to: calibration_comparison.jpg")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this module:")
        print("1. Place chessboard calibration images in a folder")
        print("2. Update 'calibration_folder' path above")
        print("3. Run: python calibration.py")
