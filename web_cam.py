import cv2
from face_detection import haar

cap = cv2.VideoCapture(0)

face_det = haar.Haar()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    face_det.detect(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()