import cv2
from face_detector import face_detector
from eye_detector import eye_detector
from eye_shape_detector import eye_shape_detector
from eye_state_detector import eye_state_detector
from data_manager import utils
import timeit

class BlinkDetector:

    # для видео используется ифнормация предыдущих кадров
    is_video = True

    face_det = None
    eye_det = None
    eye_shape_det = None
    eye_state_det = None

    def __init__(self, is_video = True):
        self.is_video = is_video
        self.face_det = face_detector.FaceDetector(is_video)
        self.eye_det = eye_detector.EyeDetector(is_video)
        self.eye_shape_det = eye_shape_detector.EyeShapeDetector()
        self.eye_state_det = eye_state_detector.EyeStateDetector()


    def detect(self, src_img):
        start_time = timeit.default_timer()

        l_res, r_res = self.detect_impl(src_img)

        height, width, chanel = src_img.shape

        elapsed = timeit.default_timer() - start_time
        if self.is_video:
            cv2.putText(src_img, 'FPS: ' + str(int(1.0 / elapsed)), (width - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

        return l_res, r_res

    # выделяет на входной картинке лицо и глаза, отмечает если моргнули
    def detect_impl(self, src_img):
        src_h, src_w, _ = src_img.shape

        src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        # лицо можно детектировать и на мелкой картинке
        face_find_img = utils.resize_img_to(src_img_gray, 200)

        face_rect = self.face_det.get_face(src_img, face_find_img, [[0, 0], [src_w, src_h]])
        if face_rect is None:
            return None, None

        face_img = utils.crop_img_by_rect(src_img_gray, face_rect)

        # меняем размер лица к 100x100
        face_img = utils.resize_img_to(face_img, 100)

        l_eye_rect, r_eye_rect = self.eye_det.get_eyes(src_img, face_img, face_rect)

        eye_img_size = 24

        l_norm_eye_rect = None
        r_norm_eye_rect = None
        l_norm_eye_img = None
        r_norm_eye_img = None

        if l_eye_rect:
            l_eye_img = utils.crop_img_by_rect(src_img_gray, l_eye_rect)
            l_eye_img = utils.resize_img_to(l_eye_img, eye_img_size)
            l_norm_eye_rect = self.eye_shape_det.get_shape(src_img, l_eye_img, l_eye_rect)
            l_norm_eye_img = utils.crop_img_by_rect(src_img_gray, l_norm_eye_rect)

        if r_eye_rect:
            r_eye_img = utils.crop_img_by_rect(src_img_gray, r_eye_rect)
            r_eye_img = utils.resize_img_to(r_eye_img, eye_img_size)
            r_norm_eye_rect = self.eye_shape_det.get_shape(src_img, r_eye_img, r_eye_rect)
            r_norm_eye_img = utils.crop_img_by_rect(src_img_gray, r_norm_eye_rect)

        if l_norm_eye_rect and r_norm_eye_rect:
            # делаем поворот
            l_c = utils.get_rect_center(l_norm_eye_rect)
            r_c = utils.get_rect_center(r_norm_eye_rect)
            angle = utils.get_v_angle([r_c, l_c])
            l_norm_eye_img = utils.rotate_img(l_norm_eye_img, angle)
            r_norm_eye_img = utils.rotate_img(r_norm_eye_img, angle)


        l_res_img = None
        r_res_img = None
        if l_norm_eye_rect:
            if self.is_video:
                cv2.imshow('l_eye', l_norm_eye_img)
            l_is_opened, l_res_img = self.eye_state_det.get_eye_state(src_img, l_norm_eye_img, l_norm_eye_rect)
        if r_norm_eye_rect:
            if self.is_video:
                cv2.imshow('r_eye', r_norm_eye_img)
            r_is_opened, r_res_img = self.eye_state_det.get_eye_state(src_img, r_norm_eye_img, r_norm_eye_rect)


        return l_res_img, r_res_img

