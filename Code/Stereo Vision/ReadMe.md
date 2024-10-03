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

## Depth Estimation
1) Simple Stereo (**[depth_estimation_simple_stereo](./depth_estimation_simple_stereo.py)**)
    - Running the file generates the disparity map
    - I downloaded [Left.png](Left.png) and [Right.png](Right.png) from some website that didn't list the camera parameters. So, I can only compute disparity map for these images.
    - One can compute the depth $D$ of the object/pixel using the disparity. $$D = \frac{b f m_x}{u_l - u_r},$$ where $b$ is the baseline of the set of stereo cameras, $f$ is the focal length of the camera (both cameras are assumed to have the same focal length), and $m_x$ is the pixel density of the image sensor in the $x$ direction, and $(u_l - u_r)$ is the disparity

2) Uncalibrated Stereo (**[depth_estimation_uncalibrated_stereo](./depth_estimation_uncalibrated_stereo.py)**)
    - Under Development