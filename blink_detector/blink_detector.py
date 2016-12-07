import cv2
from face_detector import face_detector
from eye_detector import eye_detector
from eye_shape_detector import eye_shape_detector
from data_manager import utils
import timeit

class BlinkDetector:

    # для видео используется ифнормация предыдущих кадров
    is_video = True

    face_det = None
    eye_det = None
    eye_shape_det = None

    def __init__(self, is_video = True):
        self.is_video = is_video
        self.face_det = face_detector.FaceDetector(is_video)
        self.eye_det = eye_detector.EyeDetector(is_video)
        self.eye_shape_det = eye_shape_detector.EyeShapeDetector()


    def detect(self, src_img):
        start_time = timeit.default_timer()

        self.detect_impl(src_img)

        height, width, chanel = src_img.shape

        elapsed = timeit.default_timer() - start_time
        cv2.putText(src_img, 'FPS: ' + str(int(1.0 / elapsed)), (width - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

    # выделяет на входной картинке лицо и глаза, отмечает если моргнули
    def detect_impl(self, src_img):

        face_rect, face_img = self.face_det.get_face(src_img)
        if face_img is None:
            return False

        face_h, face_w, face_chanel = face_img.shape

        # меняем размер лица к 100x100
        new_size = 100
        transform_factor = new_size / max(face_w, face_h)
        face_img = cv2.resize(face_img, (0, 0), fx=transform_factor, fy=transform_factor)

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        l_eye_img, r_eye_img, l_eye_rect, r_eye_rect = self.eye_det.get_eyes(src_img, face_img, face_rect)

        if l_eye_rect:
            l_eye_rect = utils.tuple_to_rect(l_eye_rect)
            norm_l_eye_img, norm_l_eye_rect = self.eye_shape_det.get_shape(src_img, l_eye_img, l_eye_rect)

        if r_eye_rect:
            r_eye_rect = utils.tuple_to_rect(r_eye_rect)
            norm_r_eye_img, norm_r_eye_rect = self.eye_shape_det.get_shape(src_img, r_eye_img, r_eye_rect)

        return True
