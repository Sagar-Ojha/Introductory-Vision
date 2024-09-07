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
    blurred_mat = np.zeros((len(img_mat), len(img_mat[0])), dtype=np.uint8)

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
                            len(original_mat[0]) + extension * 2), dtype=np.uint8)

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
def gradient(filter_x, filter_y, original_image):
    filter_size = len(filter_x) # Need to have same filter size for both x and y gradients
    image = copy_boundary(original_image, int(filter_size / 2))
    gradient_image = np.zeros((len(original_image), len(original_image[0]), 2),dtype=np.uint8)
    grad_x = np.zeros((len(image), len(image[0])),dtype=np.float32)
    grad_y = np.zeros((len(image), len(image[0])),dtype=np.float32)

    for i in range(int(filter_size / 2), len(image) - int(filter_size / 2)):
        for j in range(int(filter_size / 2), len(image[0]) - int(filter_size / 2)):
            grad_x[i][j] = convolution(filter_x, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            grad_y[i][j] = convolution(filter_y, \
                        image[(i - int(filter_size / 2)): (i + int(filter_size / 2) + 1),\
                              (j - int(filter_size / 2)): (j + int(filter_size / 2) + 1)])
            gradient_image[i-int(filter_size/2)][j-int(filter_size/2)][0] = \
                int(np.sqrt((grad_x[i][j])**2 + (grad_y[i][j])**2))
            gradient_image[i-int(filter_size/2)][j-int(filter_size/2)][1] = \
                int(np.arctan2(grad_y[i][j], grad_x[i][j]))

    return gradient_image
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
    sobel_filter_y = np.array([[ 1,  2,  1],
                               [ 0,  0,  0],
                               [-1, -2, -1]])

    gradient_image = gradient(sobel_filter_x, sobel_filter_y, blurred_img)
    cv2.imshow('Blurred Image', cv2.resize(blurred_img, (550, 550)))
    cv2.imshow('Gradient Image', cv2.resize(gradient_image[:,:,0], (550, 550)))

    # print(len(blurred_mat), len(blurred_mat[0]))
    # print(len(chess_grayscale), len(chess_grayscale[0]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()