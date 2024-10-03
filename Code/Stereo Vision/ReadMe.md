# Depth using Stereo

## Camera Calibration
1) Direct Linear Transformation (**[camera_calibration_DLT.py](./camera_calibration_DLT.py)**)
    - Calibrates the camera using a known 3D geometry such as a checkered cube
    - A minimum of 6 points is necessary for the calibration
    - Provide the coordinates of the point in the camera coordinate system as well as the pixel coordinate system
        - coordinates_in_camera_frame
        - coordinates_in_pixel

2) Zhang's Method (**[camera_calibration_Zhang.py](./camera_calibration_Zhang.py)**)
    - Calibrates the camera using a minimum of 3 images taken at sufficiently different depths and orientation of a known planar surface such as a checkerboard
    - Store the checkboard images for calibration inside [this](./CalibrationImages/Checkerboards/) directory
    - Setup the parameters used in *cv2.findChessboardCorners()* inside *get_intrinsic_matrix_Zhang()*
        - column_corners: number of corners along the column of the checkerboard
        - row_corners: number of corners along the row of the checkerboard
        - length_of_square: in millimeters
    - Setup the parameters used for RANSAC inside *get_intrinsic_matrix_Zhang()* as well
        - RANSAC_max_iterations
        - RANSAC_inlier_error_threshold
    - Setup the parameters using in homography computation inside *get_homography()*
        - num_correspondence_points: number of points to consider to compute homography (using least-squared method)
    - *calibrateCamera()* in openCV implements Zhang's method

## Under development
    As of now, only disparity map can be obtained. Depth estimation is under development.