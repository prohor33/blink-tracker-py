import cv2

class FaceDetector:
    # для видео используется ифнормация предыдущих кадров
    is_video = True

    def __init__(self, is_video = True):
        self.is_video = is_video

    face_cascade = cv2.CascadeClassifier('data/haars/haarcascade_frontalface_default.xml')

    cv2.ocl.setUseOpenCL(False)

    face_scale_factor = 1.1
    face_min_neighbors = 5

    # детектируем лицо
    def get_face(self, src_img):

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.face_scale_factor, self.face_min_neighbors)
        for (x,y,w,h) in faces:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # возвращаем тупо первый вариант
            roi = src_img[y:y + h, x:x + w]
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            return [(x, y), (w, h)], roi

        return None, None


