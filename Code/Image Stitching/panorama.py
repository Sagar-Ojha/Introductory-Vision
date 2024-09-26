import random
import time
import cv2
import numpy as np
import copy
#===================================================================
#===================================================================


#===================================================================
def get_matched_keypoints(image1_keypoints, image1_descriptors,\
                          image2_keypoints, image2_descriptors):
    """ Returns an array of array of tuples of matched keypoints """
    # Use the SIFT descriptor to match between the keypoints
    all_matched_keypoints = []
    for i in range(len(image1_descriptors)):
        # Lowe's ratio can be used to reject flase positives by keeping track of 2 similar matches so far
        # Checking for false positives is extremely important
        threshold_ratio = 0.7   # Lowe's ratio
        closest_match_index = 0
        closest_match_error = float('inf')
        second_closest_match_error = float('inf')

        for j in range(len(image2_descriptors)):
            error = np.linalg.norm(image1_descriptors[i] - image2_descriptors[j])

            # If the error is smaller than the current smallest error,
            # then make second error the previous smallest and update the smallest error
            if (error < closest_match_error):
                second_closest_match_error = closest_match_error
                closest_match_index = j
                closest_match_error = error
            
            # If the error is only smaller than the second error, then update the second error only
            elif (error < second_closest_match_error):
                second_closest_match_error = error

        # Obtain the matching keypoints using Lowe's ratio and append to the list of all matched keypoints
        # If the closest match is significantly better than the second closest match,
        # then the closest match is considered truly to be the closest match
        if (closest_match_error < (threshold_ratio * second_closest_match_error)):
            matched_keypoints = [image1_keypoints[i], image2_keypoints[closest_match_index]]
            all_matched_keypoints.append(matched_keypoints)

    return all_matched_keypoints
#===================================================================

#===================================================================
def matched_keypoints_intensities(all_matched_keypoints, image2_grayscale, image1_grayscale):
    """ Prints the intensities of the matched keypoints """
    # The indices of the rows and columns for grayscale images have to be carefully considered
    for i in range(len(all_matched_keypoints)):
        print(f'{image1_grayscale[int(all_matched_keypoints[i][0][1])][int(all_matched_keypoints[i][0][0])]}, '\
              f'{image2_grayscale[int(all_matched_keypoints[i][1][1])][int(all_matched_keypoints[i][1][0])]} \n')
    return
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
def warp_and_stitch(image1, image2, homography):
    """ Warps image1 and then stitches the warped image1 with image2 """
    # Get the transformed locations of the 4 corners of image1 after applying homography
    # In homogeneous coordinates
    left_top_homo = homography @ np.array([[0], [0], [1]])
    left_bottom_homo = homography @ np.array([[0], [len(image1)], [1]])
    right_top_homo = homography @ np.array([[len(image1[0])], [0], [1]])
    right_bottom_homo = homography @ np.array([[len(image1[0])], [len(image1)], [1]])

    # In pixel coordinates
    left_top = np.array([[left_top_homo[0][0]], [left_top_homo[1][0]]]) / left_top_homo[2][0]
    left_bottom = np.array([[left_bottom_homo[0][0]], [left_bottom_homo[1][0]]]) / left_bottom_homo[2][0]
    right_top = np.array([[right_top_homo[0][0]], [right_top_homo[1][0]]]) / right_top_homo[2][0]
    right_bottom = np.array([[right_bottom_homo[0][0]], [right_bottom_homo[1][0]]]) / right_bottom_homo[2][0]

    # Construct the empty canvas
    additional_left_shift = int(np.abs(min(left_top[0][0], left_bottom[0][0])))
    additional_height = int(min(left_top[1][0], right_top[1][0]))
    additional_depth  = int(max(left_bottom[1][0], right_bottom[1][0]) - len(image2))

    # Convert the height and depth values to positive integers
    if (additional_height > 0): additional_height = 0
    else: additional_height = np.abs(additional_height)
    if (additional_depth < 0): additional_depth = 0

    stitched_image = np.zeros((len(image2) + additional_height + additional_depth,\
                               len(image2[0]) + additional_left_shift, 3), dtype=np.uint8)

    # print(f'Added Height: {additional_height}')
    # print(f'Added Depth: {additional_depth}')
    # print(f'Added Left Shift: {additional_left_shift}')
    # print(f'{np.shape(stitched_image)}')

    # Loop over the canvas and get the pixel from image1
    for row in range(len(stitched_image)):
        for col in range(len(stitched_image[0])):
            # Express (col, row) in the coordinate frame of image2
            x = col - additional_left_shift
            y = row - additional_height
            image1_transformed_coordinates = np.array([[x], [y], [1]])

            # Compute the coordinates in image1 before the homography
            corresponding_image1_coordinates = np.linalg.inv(homography) @ image1_transformed_coordinates
            corresponding_image1_coordinates /= corresponding_image1_coordinates[2][0]
            x_corresponding = int(corresponding_image1_coordinates[0][0])
            y_corresponding = int(corresponding_image1_coordinates[1][0])

            # Grab the pixel from image1 and put it in stitched_image
            if ((0 <= x_corresponding < len(image1[0])) and\
                (0 <= y_corresponding < len(image1))):
                stitched_image[row][col][:] = image1[y_corresponding][x_corresponding]
    
    # Loop over the canvas one more time and get the pixel from image2
    for row in range(len(stitched_image)):
        for col in range(len(stitched_image[0])):
            # Get the pixel location in image1
            x = col - additional_left_shift
            y = row - additional_height

            # Grab the pixel from image2 and put it in stitched_image
            if ((0 <= x < len(image2[0])) and\
                (0 <= y < len(image2))):
                stitched_image[row][col][:] = image2[y][x]

    return stitched_image
#===================================================================

#===================================================================
def create_panorama(images):
    """ Returns a panorama image by combining elements of images """
    # Initialize image_left as the first element of images
    image_left = copy.deepcopy(images[0])

    # Initialize panorama
    panorama = np.array((1,1))

    # Loop over to get panorama
    for i in range(1, len(images)):
        image_right = images[i]
        image_left_grayscale = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        image_right_grayscale = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

        # Get the SIFT features from the images
        image_left_sift = cv2.SIFT_create()
        image_right_sift = cv2.SIFT_create()

        image_left_sift_keypoints, image_left_sift_descriptor = \
            image_left_sift.detectAndCompute(image_left_grayscale, None)
        image_right_sift_keypoints, image_right_sift_descriptor = \
            image_right_sift.detectAndCompute(image_right_grayscale, None)

        # Draw the keypoints along with the scale and orientation of the keypoint to the image
        # image1_with_keypoints = cv2.drawKeypoints(image1_grayscale, image1_sift_keypoints,\
        #                                           image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # image2_with_keypoints = cv2.drawKeypoints(image2_grayscale, image2_sift_keypoints,\
        #                                           image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Convert the keypoint coordinates to float
        image_left_keypoints_coordinates = cv2.KeyPoint.convert(image_left_sift_keypoints)
        image_right_keypoints_coordinates = cv2.KeyPoint.convert(image_right_sift_keypoints)

        # Match the keypoints from one image to another image
        all_matched_keypoints = get_matched_keypoints(image_left_keypoints_coordinates, image_left_sift_descriptor,\
                                                      image_right_keypoints_coordinates, image_right_sift_descriptor)
        
        # np.savetxt('all_matched_keypoints', all_matched_keypoints[0], delimiter=' ')
        # np.savetxt('image1_keypoints_coordinates', image1_keypoints_coordinates, delimiter=' ')
        # np.savetxt('image2_keypoints_coordinates', image2_keypoints_coordinates, delimiter=' ')

        # Apply RANSAC to get the Homography
        RANSAC_max_iterations = 1000
        RANSAC_inlier_error_threshold = 5
        homography = get_homography(all_matched_keypoints, RANSAC_max_iterations, RANSAC_inlier_error_threshold)
        # print(f'Homography:\n{homography}')

        # Warp image1 onto image2 and stitch the images
        stitched_image = warp_and_stitch(image_left, image_right, homography)

        # Set image1 as stitched_image for the next loop
        image_left = copy.deepcopy(stitched_image)

        # Set new panorama
        panorama = stitched_image

    return panorama
#===================================================================

#===================================================================
if __name__ == "__main__":
    initial_time = time.time()
    image1 = cv2.imread('.\Mountain1.jpg')
    image2 = cv2.imread('.\Mountain2.jpg')
    image3 = cv2.imread('.\Mountain3.jpg')
    # print(f'Shape of image1: {np.shape(image1)}')

    images = [image1, image2, image3]

    panorama = create_panorama(images)

    # Runtime
    current_time = time.time()
    total_time = current_time - initial_time
    # print(f'Number of keypoints: {len(image1_keypoints_coordinates)}')
    print(f'Total time: {total_time} s')

    # Stitched Image
    cv2.imwrite('StitchedImage.png', panorama)
    cv2.imshow('StitchedImage', panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()