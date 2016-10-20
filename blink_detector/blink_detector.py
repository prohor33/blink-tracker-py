import cv2
from face_detector import face_detector
from eye_detector import eye_detector

class BlinkDetector:

    # для видео используется ифнормация предыдущих кадров
    is_video = True

    face_det = None
    eye_det = None

    def __init__(self, is_video = True):
        self.is_video = is_video
        self.face_det = face_detector.FaceDetector(is_video)
        self.eye_det = eye_detector.EyeDetector(is_video)

    # выделяет на входной картинке лицо и глаза, отмечает если моргнули
    def detect(self, src_img):
        face_rect, face_img = self.face_det.get_face(src_img)
        if face_img is None:
            return False

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        left_eye_img, right_eye_img = self.eye_det.get_eyes(src_img, face_img, face_rect[0])

        return True
