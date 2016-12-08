import cv2
from data_manager import utils

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
    # src_img - исходное изображения, на котором можно рисовать
    # img - уменьшенное изображение в серых тонах
    def get_face(self, src_img, img, img_rect):

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        res_rect = self.get_face_impl(img)
        if res_rect is None:
            return None
        res_rect = utils.convert_rect_to_parent(img, img_rect, res_rect)
        utils.draw_rect(src_img, res_rect, (255, 0, 0), 1)
        return res_rect

    def get_face_impl(self, img):
        faces = self.face_cascade.detectMultiScale(img, self.face_scale_factor, self.face_min_neighbors)
        for (x,y,w,h) in faces:
            # возвращаем тупо первый вариант
            return [[x, y], [w, h]]

        return None
