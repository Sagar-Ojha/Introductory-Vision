import cv2
import numpy as np
img = cv2.imread('ChessBoard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(np.float32(img))
##cv2.imshow('gray', gray)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,3,3,0.1)
dst = cv2.dilate(dst,None)
img[dst>0.035*dst.max()]=[255,255,0]

##print(gray)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()