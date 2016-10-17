import cv2
from face_detection import haar

face_det = haar.Haar()

img = cv2.imread('data/0.png')
face_det.detect(img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()