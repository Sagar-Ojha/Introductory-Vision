import cv2
import numpy as np
#===================================================================
#===================================================================

#===================================================================
def gaussian_filter(σ):
    filter_size = int(2 * np.pi * σ)
    if (filter_size % 2 == 0): filter_size = int(filter_size + 1)
    filter_kernel = np.zeros((filter_size, filter_size))

    for i in range(len(filter_kernel)):
        for j in range(len(filter_kernel[0])):
            filter_kernel[i][j] = np.exp(-((i - filter_size//2)**2 +
                                (j - filter_size//2)**2) / (2* (σ**2)))
    coefficient = 1 / (2 * np.pi * (σ**2))

    return coefficient * filter_kernel
#-------------------------------------------------------------------
def gaussian_blur(img_mat, filter_kernel):
    filter_sum = np.sum(filter_kernel)
    # extend the boundary of the original matrix using copy method
    extension = int(len(filter_kernel)/2)
    extended_mat = copy_boundary(img_mat, extension)
    blurred_mat = np.zeros((len(img_mat), len(img_mat[0])))

    # apply the filter over extended_mat
    for i in range(len(extended_mat)):
        for j in range(len(extended_mat[0])):
            if (i >= extension) and (i <= (len(blurred_mat) + extension - 1)) and\
               (j >= extension) and (j <= (len(blurred_mat[0]) + extension - 1)):
                blurred_mat[i - extension][j - extension] = \
                    (1 / (filter_sum)) * convolution(filter_kernel, \
                    extended_mat[(i - extension):(i + extension + 1), \
                                 (j - extension):(j + extension + 1)])

    return blurred_mat
#-------------------------------------------------------------------
def copy_boundary(original_mat, extension):
    extended_mat = np.zeros((len(original_mat) + extension * 2,\
                            len(original_mat[0]) + extension * 2))

    for i in range(len(extended_mat)):
        for j in range(len(extended_mat[i])):
            if ((i < extension) and (j < extension)):
                extended_mat[i][j] = original_mat[0][0]
            elif ((i > (extension + len(original_mat) - 1)) and\
                  (j > (extension + len(original_mat[0]) - 1))):
                extended_mat[i][j] = original_mat[-1][-1]
            elif ((j < extension) and (i > (len(original_mat) + extension - 1))):
                extended_mat[i][j] = original_mat[-1][0]
            elif ((i < extension) and (j > (len(original_mat[0]) + extension - 1))):
                extended_mat[i][j] = original_mat[0][-1]
            elif (i < extension):
                extended_mat[i][j] = original_mat[0][j - extension]
            elif (i > (extension + len(original_mat) - 1)):
                extended_mat[i][j] = original_mat[-1][j - extension]
            elif (j < extension):
                extended_mat[i][j] = original_mat[i - extension][0]
            elif (j > (extension + len(original_mat[0]) - 1)):
                extended_mat[i][j] = original_mat[i - extension][-1]
            else:
                extended_mat[i][j] = original_mat[i - extension][j - extension]

    return extended_mat
#-------------------------------------------------------------------
def convolution(mat1, mat2):
    new_mat = np.multiply(mat1, mat2)
    sum = np.sum(new_mat)
    return sum
#===================================================================

#===================================================================
def get_gradient_in_x_y(filter_x, filter_y, original_image):
    """ Returns gradient in both x and y directions """
    filter_size = len(filter_x) # Need to have same filter size for both x and y gradients
    image = copy_boundary(original_image, int(filter_size / 2))

    # grad_x & grad_y have extended boundaries
    grad_x = np.zeros((len(image), len(image[0])))
    grad_y = np.zeros((len(image), len(image[0])))

    # gradient_x & gradient_y have the same boundary as original_image
    gradient_image_x = np.zeros((len(original_image), len(original_image[0])))
    gradient_image_y = np.zeros((len(original_image), len(original_image[0])))

    for i in range(int(filter_size / 2), len(image) - int(filter_size / 2)):
        for j in range(int(filter_size / 2), len(image[0]) - int(filter_size / 2)):
            grad_x[i][j] = convolution(filter_x, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            grad_y[i][j] = convolution(filter_y, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            gradient_image_x[i-int(filter_size/2)][j-int(filter_size/2)] = grad_x[i][j]
            gradient_image_y[i-int(filter_size/2)][j-int(filter_size/2)] = grad_y[i][j]

    return gradient_image_x, gradient_image_y
#===================================================================

#===================================================================
def get_corner_coordinates(gradient_image_x, gradient_image_y, window_size):
    """ Returns the pixel coordinates of the corners """
    corners_coordinates = []

    image_x = copy_boundary(gradient_image_x, int(window_size / 2))
    image_y = copy_boundary(gradient_image_y, int(window_size / 2))

    for row in range(int(window_size / 2), len(image_x) - int(window_size / 2)):
        for col in range(int(window_size / 2), len(image_x[0]) - int(window_size / 2)):
            # Select the window around the pixel to detect corner
            window_grad_x = image_x[(row - int(window_size / 2)): (row + int(window_size / 2) + 1),\
                                    (col - int(window_size / 2)): (col + int(window_size / 2) + 1)]
            window_grad_y = image_y[(row - int(window_size / 2)): (row + int(window_size / 2) + 1),\
                                    (col - int(window_size / 2)): (col + int(window_size / 2) + 1)]
            
            # Correlation matrix
            correlation_matrix = compute_correlation_matrix(window_grad_x, window_grad_y)

            # Eigen values of the correlation matrix
            # eigen_value_1 = (1/2) * (correlation_matrix[0][0] + correlation_matrix[1][1] +\
            #                          ((2 * correlation_matrix[0][1])**2 +\
            #                           (correlation_matrix[0][0] - correlation_matrix[1][1])**2)**(1/2))
            # eigen_value_2 = (1/2) * (correlation_matrix[0][0] + correlation_matrix[1][1] +\
            #                          ((2 * correlation_matrix[0][1])**2 -\
            #                           (correlation_matrix[0][0] - correlation_matrix[1][1])**2)**(1/2))

            eigen_values = np.linalg.eigvals(correlation_matrix)
            eigen_value_1, eigen_value_2 = eigen_values

            # Check for the threshold and claim the pixel as a corner
            k = 0.05
            r_function = eigen_value_1 * eigen_value_2 - k * (eigen_value_1 + eigen_value_2)**2
            r_threshold = 0.9 * np.max(r_function)

            if ((r_function > r_threshold) & (not(np.isnan(r_function)))):
                corner_coordinate = np.array([row - int(window_size / 2), col - int(window_size / 2)])
                corners_coordinates.append(corner_coordinate)

    return corners_coordinates
#===================================================================

#===================================================================
def compute_correlation_matrix(gradient_x, gradient_y):
    """ Returns the correlation matrix """
    gradient_x = gradient_x - np.average(gradient_x)
    gradient_y = gradient_y - np.average(gradient_y)
    correlation_matrix_00 = np.sum(gradient_x * gradient_x)
    correlation_matrix_01 = np.sum(gradient_x * gradient_y)
    correlation_matrix_11 = np.sum(gradient_y * gradient_y)
    correlation_matrix = np.array([[correlation_matrix_00, correlation_matrix_01],\
                                   [correlation_matrix_01, correlation_matrix_11]])

    return correlation_matrix
#===================================================================

#===================================================================
def show_image_corners(corners_coordinates, original_image):
    """ Returns the image with corners highlighted """
    image_with_corner = np.array(original_image)
    for corner in corners_coordinates:
        image_with_corner[corner[0]][corner[1]][0] = 200
        image_with_corner[corner[0]][corner[1]][1] = 100
        image_with_corner[corner[0]][corner[1]][2] = 200

    return image_with_corner
#===================================================================

#===================================================================
if __name__ == "__main__":
    sample_img = cv2.imread('.\Lena.png')
    sample_grayscale = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

    σ = 1.75 # Standard deviation of the Gaussian filter
    filter_kernel = gaussian_filter(σ)
    # print(filter_kernel.round(3))
    blurred_img = gaussian_blur(sample_grayscale, filter_kernel)
    sobel_filter_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_filter_y = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])

    gradient_image_x, gradient_image_y = get_gradient_in_x_y(sobel_filter_x, sobel_filter_y, sample_grayscale)
    # print(np.max(gradient_image_x))
    # print(np.max(gradient_image_y))
    np.savetxt('gradient_x', gradient_image_x, delimiter=' ')
    np.savetxt('gradient_y', gradient_image_y, delimiter=' ')

    window_size = 5
    gradient_image_x = gaussian_blur(gradient_image_x, filter_kernel)
    gradient_image_y = gaussian_blur(gradient_image_y, filter_kernel)
    corners_coordinates = get_corner_coordinates(gradient_image_x, gradient_image_y, window_size)
    # print(corners_coordinates)

    image_with_corner = show_image_corners(corners_coordinates, sample_img)

    # dst = cv2.cornerHarris(blurred_img,3,3,0.1)
    # dst = cv2.dilate(dst,None)
    # sample_img[dst>0.035*dst.max()]=[255,255,0]

    ##print(gray)

    cv2.imshow('image',cv2.resize(sample_img, (550, 550)))

    cv2.imshow('Blurred Image', cv2.resize(blurred_img.astype(np.uint8), (550, 550)))
    cv2.imshow('Gradient Image X', cv2.resize(gradient_image_x.astype(np.uint8), (550, 550)))
    cv2.imshow('Gradient Image Y', cv2.resize(gradient_image_y.astype(np.uint8), (550, 550)))
    cv2.imshow('Corner Image', cv2.resize(image_with_corner.astype(np.uint8), (550, 550)))

    # print(len(blurred_mat), len(blurred_mat[0]))
    # print(len(chess_grayscale), len(chess_grayscale[0]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()