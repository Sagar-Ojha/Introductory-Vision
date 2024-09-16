import time
import cv2
import numpy as np
#===================================================================
#===================================================================


#===================================================================
def copy_boundary(original_mat, extension):
    """ Returns a padded/extended matrix """
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
def normalized_correlation(matrix1, matrix2):
    """ Returns the normalized correlation of 'matrix1' & 'matrix2' """
    matrix1_energy = (np.sum(np.power(matrix1, 2)))**(1/2)
    matrix2_energy = (np.sum(np.power(matrix2, 2)))**(1/2)
    correlation = np.sum(np.multiply(matrix1, matrix2)) /\
                  (matrix1_energy * matrix2_energy)
    # print(f'{normalized_matrix1}')
    # print(f'{normalized_matrix2}')
    # print(f'{correlation}')

    return correlation
#===================================================================

#===================================================================
def best_correlation(small_matrix, wide_matrix, start, end):
    """ Returns the index of 'wide_matrix' with the best correlation
        with 'small_matrix' """
    # 'start' & 'end' are the extreme indices for the correlation test
    correlation_index = start
    correlation = 0
    small_matrix_size = len(small_matrix)

    for i in range(start, end+1):
        test_window = wide_matrix[:, (i - int(small_matrix_size / 2)):\
                                  (i + int(small_matrix_size / 2) + 1)]
        current_correlation = normalized_correlation(small_matrix, test_window)

        if (correlation < current_correlation):
            correlation = current_correlation
            correlation_index = i

    return correlation_index - start
#===================================================================

#===================================================================
def scanline_matrix(original_matrix, window_size, row_index):
    """ Returns the 'wide_matrix' of height 'window_size' at row 'index' """
    wide_matrix = original_matrix[(row_index - int(window_size / 2)):\
                            (row_index + int(window_size / 2) + 1), :]

    return wide_matrix
#===================================================================

#===================================================================
def disparity_image(left_image, right_image, window_size):
    """ Get the locations of the pixels of the right image corresponding
        to the pixel location of the left image """
    disparity = np.zeros((len(left_image), len(left_image[0])))

    left_image_padded = copy_boundary(left_image, int(window_size / 2))
    right_image_padded = copy_boundary(right_image, int(window_size / 2))
    # print(f'{left_image_padded}')
    # print(f'{right_image_padded}')

    # 'start' & 'end' are the extremes of indices to scan the 'wide_matrix'
    start = int(window_size / 2)
    end = len(left_image_padded[0]) - int(window_size / 2) - 1

    for row in range(int(window_size / 2), len(left_image_padded) - int(window_size / 2)):
        wide_matrix = scanline_matrix(right_image_padded, window_size, row)
        # print(f'{wide_matrix}')
        print(row)
        for col in range(int(window_size / 2), len(left_image_padded[0]) - int(window_size / 2)):
            small_matrix = left_image_padded[(row - int(window_size / 2)): (row + int(window_size / 2) + 1),\
                                             (col - int(window_size / 2)): (col + int(window_size / 2) + 1)]
            # print(f'{small_matrix}')

            correlation_column_index = best_correlation(small_matrix, wide_matrix, start, end)
            # print(f'{correlation_column_index}')

            # Disparity between the pixel index/location of the left and right images
            if (col != correlation_column_index):
                disparity[row - int(window_size / 2)][col - int(window_size / 2)] =\
                    1 / (col - correlation_column_index)
            # TODO: Check for redundancy
            else:
                disparity[row - int(window_size / 2)][col - int(window_size / 2)] = 0.5

    return disparity
#===================================================================

#===================================================================
def map_to_range(image, range):
    """ Maps the values in 'image' to the 'range' """
    min_value = np.min(image)
    max_value = np.max(image)

    slope = (range[1] - range[0]) / (max_value - min_value)
    mapped_image = slope * (image - min_value)
    return mapped_image.astype(np.uint8)
#===================================================================

#===================================================================
if __name__ == "__main__":
    start = time.time()
    left_image = cv2.imread('.\Left.png')
    right_image = cv2.imread('.\Right.png')

    left_grayscale = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_grayscale = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    window_size = 3

    # Test
    # left_grayscale = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                            [1, 0, 4, 6, 7, 2, 5, 7, 9],
    #                            [60, 58, 60, 600, 58, 36, 57, 188, 209],
    #                            [17, 2, 23, 46, 5, 56, 7, 8, 10],
    #                            [1, 200, 143, 164, 255, 206, 76, 80, 99]])
    
    # right_grayscale = np.array([[3, 4, 5, 6, 7, 8, 9, 78, 90],
    #                            [6, 7, 2, 5, 7, 9, 52, 14, 22],
    #                            [60, 600, 58, 36, 57, 188, 209,89,90],
    #                            [46, 5, 56, 7, 8, 10, 89, 56, 67],
    #                            [143, 164, 255, 206, 76, 80, 99, 23, 44]])

    disparity = disparity_image(left_grayscale, right_grayscale, window_size)
    mapped_disparity = map_to_range(disparity, np.array([0, 255]))

    cv2.imwrite('DisparityImage.jpg', mapped_disparity)
    # cv2.imshow('Disparity Image', cv2.resize(mapped_disparity, (550, 550)))
    print(f'Done in: {time.time() - start} s.')

    cv2.waitKey(0)
    cv2.destroyAllWindows()