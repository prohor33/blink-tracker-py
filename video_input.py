import cv2
import time
from face_detection import haar

cap = cv2.VideoCapture('data/video0.mp4')

face_det = haar.Haar()

while(True):
    cap.grab()

    # Capture frame-by-frame
    ret, frame = cap.retrieve()

    if not ret:
        print("error: no frame")
        break

    face_det.detect(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.2)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()