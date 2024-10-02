import numpy as np
from scipy.linalg import rq
#===================================================================
#===================================================================

#===================================================================
def get_projection_matrix_DLT(coordinates_in_camera_frame, coordinates_in_pixel):
    """ Returns the projection matrix after calibrating the camera """
    # Loop over the corners and formulate the world coordinates of the corner
    # Also, create the A matrix
    A = np.empty(shape=[0, 12])
    for i in range(len(coordinates_in_camera_frame)):
        # Point pixel coordinates
        u = coordinates_in_pixel[i][0]
        v = coordinates_in_pixel[i][1]

        # Point w.r.t. camera coordinates
        x = coordinates_in_camera_frame[i][0]
        y = coordinates_in_camera_frame[i][1]
        z = coordinates_in_camera_frame[i][2]
        # print(f'(u, v): ({u}, {v})')
        # print(f'(x, y, z): ({x}, {y}, {z})')

        # Formulating A matrix
        a1 = np.array([[x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u]])
        a2 = np.array([[0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]])
        a = np.append(a1, a2, axis=0)
        A = np.append(A, a, axis=0)

    # One could get p using eigenvector with the smallest eigenvalue of A^T A
    # but the eigen method results in erronous p
    # p vector can be obtained from the SVD of A
    U, S, Vh = np.linalg.svd(A)
    p = Vh[-1]
    # print(p)

    projection_matrix = np.array([[p[0], p[1], p[2], p[3]],
                                  [p[4], p[5], p[6], p[7]],
                                  [p[8], p[9], p[10],p[11]]])

    # Testing the reprojection
    # reproject(projection_matrix, coordinates_in_camera_frame)

    return projection_matrix
#===================================================================

#===================================================================
def reproject(projection_matrix, coordinates_in_camera_frame):
    """ Prints the (u,v) coordinates of the points after projecting coordinates_in_camera_frame"""
    for point in coordinates_in_camera_frame:
        point_homo = np.array([[point[0]], [point[1]], [point[2]], [1]])
        pixel_homo = projection_matrix @ point_homo

        u = pixel_homo[0] / pixel_homo[2]
        v = pixel_homo[1] / pixel_homo[2]
        print(f'(u, v): ({u}, {v})')
    return
#===================================================================

#===================================================================
def get_intrinsic_matrix(projection_matrix):
    """ Returns the intrinsic matrix given the projection matrix """
    projection_matrix_3_by_3 = projection_matrix[0:3, 0:3]

    # Perform RQ factorization rather than QR because
    # the upper triangular (R) comes before the orthogonal matrix (Q)
    # for the projection matrix factorization
    intrinsic_matrix, _ = rq(projection_matrix_3_by_3)
    intrinsic_matrix /= intrinsic_matrix[2][2]

    # Set the skew metrics, i.e. intrinsic_matrix[0][1] to 0 manually
    intrinsic_matrix[0][1] = 0

    return intrinsic_matrix
#===================================================================

#===================================================================
if __name__ == "__main__":
    # We obtain the coordinates of the point in camera coordinates and the pixel coordinates manually
    square_length = 22.5  # Length of the unit checkered pattern in the cube in millimeters
    coordinates_in_camera_frame = np.array([[1,0,1],
                                            [0,0,2],
                                            [0,1,2],
                                            [0,2,1],
                                            [0,2,2],
                                            [2,0,3],
                                            [1,1,3],
                                            [2,1,3],
                                            [1,2,3],
                                            [2,2,3]]) * square_length
    coordinates_in_pixel = np.array([[309,340],
                                     [288,347],
                                     [230,322],
                                     [180,332],
                                     [177,295],
                                     [352,221],
                                     [267,234],
                                     [295,198],
                                     [211,209],
                                     [243,177]])

    # Use the DLT method to calibrate the camera
    projection_matrix = get_projection_matrix_DLT(coordinates_in_camera_frame, coordinates_in_pixel)
    intrinsic_matrix = get_intrinsic_matrix(projection_matrix)

    print(f'Intrinsic Matrix:\n{intrinsic_matrix}')