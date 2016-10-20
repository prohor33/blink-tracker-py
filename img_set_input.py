import cv2
from face_detector import face_detector

face_det = face_detector.Haar()
i = 0

while True:
    img = cv2.imread('data/img_set0/' + str(i) + '.png')
    if img is None:
        break
    face_det.detect(img)
    cv2.imshow('img' + str(i), img)
    print('window' + str(i))
    i += 1

cv2.waitKey(0)
cv2.destroyAllWindows()