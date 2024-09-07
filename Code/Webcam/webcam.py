import cv2
s = cv2.VideoCapture(0)
while(1):
    ret,frame = s.read()
    print(ret)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k ==27:
        break
s.release()
cv2.destroyAllWindows()