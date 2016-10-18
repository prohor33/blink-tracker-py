import cv2

class Haar:
    """A simple haar face/eye detection"""

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

    lefteye_2splits_cascade = cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')
    righteye_2splits_cascade = cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')
    eyeglasses_cascade = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')

    scale_factor = 4.0  # video0
    # scale_factor = 3.0  # video1
    min_neighbors = 1

    cv2.ocl.setUseOpenCL(False)

    def detect(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # left eye
            # red
            eyes = self.lefteye_2splits_cascade.detectMultiScale(roi_gray, self.scale_factor, self.min_neighbors)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

            # right eye
            # pink
            eyes2 = self.righteye_2splits_cascade.detectMultiScale(roi_gray, self.scale_factor, self.min_neighbors)
            for (ex,ey,ew,eh) in eyes2:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(137,66,244),2)

            # both eyes
            # green
            eyes3 = self.eyeglasses_cascade.detectMultiScale(roi_gray, self.scale_factor, self.min_neighbors)
            for (ex, ey, ew, eh) in eyes3:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


            # both eyes simple
            # orange
            eyes4 = self.eye_cascade.detectMultiScale(roi_gray, self.scale_factor, self.min_neighbors)
            for (ex, ey, ew, eh) in eyes4:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (66, 244, 244), 2)


