import random
import time
import cv2
import numpy as np
import glob     # for reading files given the path
#===================================================================
#===================================================================

#===================================================================
def get_homography(all_matched_points, RANSAC_max_iterations, RANSAC_inlier_error_threshold):
    """ Applies RANSAC and obtains the homography """
    best_homography = np.zeros((3,3))
    num_correspondence_points = 25  # Homography will be the best fit for these correspondence points
    matches = all_matched_points

    inliers_for_best_model = 0
    # Apply RANSAC to get the best homography
    for i in range(RANSAC_max_iterations):
        rands = random.sample(range(len(all_matched_points)), num_correspondence_points)

        # Construct A matrix from the source and destination image pixel coordinates
        # The indices of the rows and columns for grayscale images have to be carefully considered
        A = np.empty(shape=[0, 9])
        for r in rands:
            a1 = np.array([[matches[r][0][0], matches[r][0][1], 1, 0, 0, 0, -matches[r][1][0] * matches[r][0][0],\
                           -matches[r][1][0] * matches[r][0][1], -matches[r][1][0]]])
            a2 = np.array([[0, 0, 0, matches[r][0][0], matches[r][0][1], 1, -matches[r][1][1] * matches[r][0][0],\
                           -matches[r][1][1] * matches[r][0][1], -matches[r][1][1]]])
            a = np.append(a1, a2, axis=0)
            A = np.append(A, a, axis=0)

        # The homography vector is the eigenvector of A^T A corresponding to the minimum eigenvalue
        # eigenvalues, eigenvectors = np.linalg.eig(np.transpose(A) @ A)
        # h = eigenvectors[np.argmin(eigenvalues)]
        
        # Alternatively, homography can be computed using SVD as well
        U, S, Vh = np.linalg.svd(A)
        h = Vh[-1]

        # Construct the matrix
        current_homography = np.array([[h[0], h[1], h[2]],
                                       [h[3], h[4], h[5]],
                                       [h[6], h[7], h[8]]]) / h[8]

        # Check the number of inliers produced by the given estimate of the homography
        current_inliers = 0
        for j in range(len(matches)):
            source_pixel_position = np.array([[matches[j][0][0]], [matches[j][0][1]], [1]])
            destination_pixel_position = np.array([[matches[j][1][0]], [matches[j][1][1]]])
            # print(destination_pixel_position)
            destination_pixel_position_prediction = current_homography.dot(source_pixel_position)
            # print(f'before dividing by homogeneous: {destination_pixel_position_prediction}')

            # Making predicted position homogeneous
            z_d = destination_pixel_position_prediction[2][0]
            destination_pixel_position_prediction = np.array([[destination_pixel_position_prediction[0][0]],
                                                              [destination_pixel_position_prediction[1][0]]])
            destination_pixel_position_prediction /= z_d

            # Check if the predicted position is within the threshold distance from the actual position
            if (np.linalg.norm(destination_pixel_position_prediction - destination_pixel_position) <\
                RANSAC_inlier_error_threshold):
                current_inliers += 1
                # print(f'{current_inliers}')

        if (current_inliers > inliers_for_best_model):
            inliers_for_best_model = current_inliers
            best_homography = current_homography
            # print(f'Homography: \n{best_homography}')
            # print(f'Inliers: {inliers_for_best_model}')

    return best_homography
#===================================================================

#===================================================================
def match_checkerboard_coordinates(grayscale_image, column_corners, row_corners, length_of_square):
    """ Returns the array of array of the (x,y) coordinates of the corner
    in the world coordinates and the pixel coordinates """
    # We know the coordinates of the corners of the checkboard
    # Therefore, there is no need to use a SIFT detector or any corner detectors

    # Get the corners
    _, corners_coordinates = cv2.findChessboardCorners(grayscale_image, (column_corners, row_corners), None)

    # Match the image and the world coordinates
    all_matched_points = []

    # Loop over the corners and formulate the world coordinates of the corner
    for i in range(len(corners_coordinates)):
        # Corner pixel coordinates
        u = int(corners_coordinates[i][0][0])
        v = int(corners_coordinates[i][0][1])

        # Corner w.r.t. camera coordinates (Only (x,y) is needed for homography)
        x = (int(i % column_corners) + 1) * length_of_square
        y = (int(i / column_corners) + 1) * length_of_square
        # print(f'(u, v): ({u}, {v})')
        # print(f'(x, y): ({x}, {y})')

        # Append to all_matched_points
        all_matched_points.append([[x,y], [u,v]])

    return all_matched_points
#===================================================================

#===================================================================
def get_intrinsic_matrix_Zhang(grayscale_images):
    """ Returns the projection matrix after calibrating the camera """
    # Set the number of corners along the row and column
    column_corners = 10
    row_corners = 4
    # One square is 32 mm long in the world coordinates
    length_of_square = 32

    # For RANSAC
    RANSAC_max_iterations = 1000
    RANSAC_inlier_error_threshold = 2

    V = np.empty(shape=[0, 6])
    # Loop over all the images to construct the V matrix
    for i in range(len(grayscale_images)):
        # print(f'Image index: {i}')
        # Match the points and compute the homography for the image
        all_matched_points = match_checkerboard_coordinates(grayscale_images[i],\
                                    column_corners, row_corners, length_of_square)
        h = get_homography(all_matched_points, RANSAC_max_iterations,\
                           RANSAC_inlier_error_threshold)
        # print(f'Homography: {h}')

        # Construct V using a minimum of 3 images
        v11 = np.array([[h[0][0] * h[0][0], h[0][0] * h[1][0] + h[1][0] * h[0][0], h[2][0] * h[0][0] + h[0][0] * h[2][0],\
                         h[1][0] * h[1][0], h[1][0] * h[2][0] + h[2][0] * h[1][0], h[2][0] * h[2][0]]])
        v12 = np.array([[h[0][0] * h[0][1], h[0][0] * h[1][1] + h[1][0] * h[0][1], h[2][0] * h[0][1] + h[0][0] * h[2][1],\
                         h[1][0] * h[1][1], h[1][0] * h[2][1] + h[2][0] * h[1][1], h[2][0] * h[2][1]]])
        v22 = np.array([[h[0][1] * h[0][1], h[0][1] * h[1][1] + h[1][1] * h[0][1], h[2][1] * h[0][1] + h[0][1] * h[2][1],\
                         h[1][1] * h[1][1], h[1][1] * h[2][1] + h[2][1] * h[1][1], h[2][1] * h[2][1]]])
        v_single_image = np.append(v12, v11 - v22, axis=0)
        V = np.append(V, v_single_image, axis=0)

    # Solve the SVD of V to get B and Cholesky decomposition of B to get (K^(-1))^T
    U, S, Vh = np.linalg.svd(V)
    b = Vh[-1]
    B = np.array([[b[0], b[1], b[2]],
                  [b[1], b[3], b[4]],
                  [b[2], b[4], b[5]]]) / b[5]

    # Obtain the Cholesky decomposition of B
    K_inv_T = np.linalg.cholesky(B)
    intrinsic_matrix = np.linalg.inv(np.transpose(K_inv_T))

    return intrinsic_matrix/intrinsic_matrix[2][2]
#===================================================================

#===================================================================
if __name__ == "__main__":
    total_time = time.time()
    image_path = '.\CalibrationImages\Checkerboards\*.jpg'
    images = []

    # Read off the files in image_path and append them to the images
    for file in glob.glob(image_path):
        image = cv2.imread(file)
        images.append(image)

    # Obtain the grayscale images
    grayscale_images = []
    for image in images:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_images.append(grayscale_image)

    # Calibrate the camera (get intrinsic paramter)
    # Also, set the parameters for RANSAC and chessboard detector inside get_intrinsic_matrix_Zhang
    intrinsic_matrix = get_intrinsic_matrix_Zhang(grayscale_images)
    print(f'Intrinsic Matrix:\n{intrinsic_matrix}')

    print(f'Total time: {round(time.time() - total_time, 2)} s.')

    cv2.waitKey(0)
    cv2.destroyAllWindows()