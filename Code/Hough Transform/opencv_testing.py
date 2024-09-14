import cv2
import numpy as np
#===================================================================

#===================================================================
if __name__ == "__main__":
    sample_img = cv2.imread('.\ChessBoard.png')
    sample_grayscale = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

    # Random img
    test_img = np.array([[2*255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                         [255, 2*255, 255, 255, 255, 255, 255, 255, 255, 255],
                         [255, 255, 2*255, 255, 255, 255, 255, 255, 255, 255],
                         [255, 255, 255, 2*255, 255, 255, 255, 255, 255, 255],
                         [255, 255, 255, 255, 2*255, 255, 255, 255, 255, 255],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.uint8)

    cv2.imshow('Full Image', sample_grayscale)
    cv2.imshow('Resized Image', cv2.resize(sample_grayscale, (550, 550)))
    cv2.imshow('Test Image', test_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(len(sample_grayscale), len(sample_grayscale[0]))
    print(sample_grayscale)
    print(test_img[5])