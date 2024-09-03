import numpy as np
import cv2
from numpy.core.numeric import identity
arr = np.array([[1,2,3],[1,2,6]])
arr2 = np.array([[1,2,3],[1,2,6]])


ima1 = cv2.imread(".\Lena.png")
ima2 = cv2.cvtColor(ima1, cv2.COLOR_BGR2GRAY)
ima3 = np.array(ima2)
ima4 = ima3[0:500, 0:500]
# junk = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])

#print(np.multiply(arr,arr2))
#print(junk)

r = 510
c = 510
sob = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
junk2 = np.identity(510)
maxs = 0

for i in range(r):
    for j in range(c):
        tempmat = ima3[j:j+3, i:i+3]
        newmat = np.multiply(tempmat,sob)
        s = np.sum(newmat)
        junk2[j,i] = s
        if s>maxs:
            maxs = s



print(ima4.shape)
print(junk2)
#junk2 = np.where(junk2<0, 0, junk2)
#cv2.imshow('pic', ima2)
cv2.imshow('pic',ima3)
cv2.waitKey(0)
cv2.destroyAllWindows()