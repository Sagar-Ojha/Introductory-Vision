import cv2
import numpy as np
#===================================================================
#===================================================================

#===================================================================
def gaussian_blur(image, filter_kernel, σ):
    blurred_image = cv2.GaussianBlur(image, (filter_kernel, filter_kernel), σ)
    return blurred_image
#===================================================================

#===================================================================
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
#===================================================================

#===================================================================
def sum_squared(mat1, mat2):
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
            grad_x[i][j] = sum_squared(filter_x, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            grad_y[i][j] = sum_squared(filter_y, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            gradient_image_x[i-int(filter_size/2)][j-int(filter_size/2)] = grad_x[i][j]
            gradient_image_y[i-int(filter_size/2)][j-int(filter_size/2)] = grad_y[i][j]

    return gradient_image_x, gradient_image_y
#===================================================================

#===================================================================
def get_corner_coordinates(gradient_image_x, gradient_image_y, window_size, σ):
    """ Returns the pixel coordinates of the corners """
    corners_coordinates = []

    I_x2 = gaussian_blur(gradient_image_x * gradient_image_x, window_size, σ)
    I_y2 = gaussian_blur(gradient_image_y * gradient_image_y, window_size, σ)
    I_xy = gaussian_blur(gradient_image_x * gradient_image_y, window_size, σ)

    # The 'gaussian_blur' considers a small window around the pixel
    # Hence, we no longer have to iterate over all the pixels to get the correlation matrix
    determinant = I_x2 * I_y2 - 2 * I_xy
    trace = I_x2 + I_y2
    k = 0.05
    r_value = determinant - k * trace * trace
    r_threshold = 0.05 * np.max(r_value)
    for row in range(len(r_value)):
        for col in range(len(r_value[0])):
            if (r_value[row][col] > r_threshold):
                corners_coordinates.append([row, col])

    return corners_coordinates
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
    sample_img = cv2.imread('.\CheckerBoard.png')
    sample_grayscale = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

    σ = 1.75 # Standard deviation of the Gaussian filter
    kernel_size = 3
    # print(filter_kernel.round(3))
    blurred_img = gaussian_blur(sample_grayscale, kernel_size, σ)
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

    window_size = 3
    corners_coordinates = get_corner_coordinates(gradient_image_x, gradient_image_y, window_size, σ)
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