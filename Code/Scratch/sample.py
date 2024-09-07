import numpy as np
import cv2
import matplotlib.pyplot as plt
ima1 = cv2.imread('./Lena.png')
ima2 = cv2.cvtColor(ima1, cv2.COLOR_BGR2GRAY)
ima3 = np.array(ima2)       ##Redundant

#ima3 = ima3[175:300, 0:500]       ##Nice Capture
p = ima3.shape
c = p[1] -2
r = p[0] -2

sob = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
junk = np.random.randint(0,3,(r,c))
maxs = 0
mins = 0

for i in range(r):
    for j in range(c):
        tempmat = ima3[i:i+3, j:j+3]
        newmat = np.multiply(tempmat,sob)
        s = np.sum(newmat) / 8
        junk[i,j] = s
        if s>maxs:
            maxs = s
        if s<mins:
            mins = s


junk = junk.astype(np.float32)

####for y axis

sob = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
junk2 = np.random.randint(0,3,(r,c))
maxs = 0
mins = 0
i=0
j=0
for i in range(r):
    for j in range(c):
        tempmat = ima3[i:i+3, j:j+3]
        newmat = np.multiply(tempmat,sob)
        s = np.sum(newmat) / 8
        junk2[i,j] = s
        if s>maxs:
            maxs = s
        if s<mins:
            mins = s


junk2 = junk2.astype(np.float32)

#####for y axis
finimg = np.sqrt(np.square(junk)+np.square(junk2))
plt.imshow(finimg, cmap='gray')
plt.show()