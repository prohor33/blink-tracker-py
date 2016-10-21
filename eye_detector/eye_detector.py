import cv2

class EyeDetector:
    # для видео используется ифнормация предыдущих кадров
    is_video = True

    def __init__(self, is_video = True):
        self.is_video = is_video

    haars_fld = 'data/haars/'

    eye_cascade = cv2.CascadeClassifier(haars_fld + 'haarcascade_eye.xml')
    lefteye_2splits_cascade = cv2.CascadeClassifier(haars_fld + 'haarcascade_lefteye_2splits.xml')
    righteye_2splits_cascade = cv2.CascadeClassifier(haars_fld + 'haarcascade_righteye_2splits.xml')
    eyeglasses_cascade = cv2.CascadeClassifier(haars_fld + 'haarcascade_eye_tree_eyeglasses.xml')

    scale_factor = 1.1
    min_neighbors = 5

    cv2.ocl.setUseOpenCL(False)

    # на вход подается лицо в серых тонах
    def get_eyes(self, src_img, face_img, face_rect):

        l_eye_img = None
        l_eye_rect = None
        r_eye_img = None
        r_eye_rect = None

        # лицо в изначальной картинке
        face_x = face_rect[0][0]
        face_y = face_rect[0][1]
        face_w = face_rect[1][0]
        face_h = face_rect[1][1]

        height, width = face_img.shape

        to_src_coord = lambda x, y, w, h: (int(face_x + x / width * face_w), int(face_y + y / height * face_h),
                                           int(w / width * face_w), int(h / height * face_h))

        # # both eyes
        # # green
        # eyes3 = self.eyeglasses_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        # for (ex, ey, ew, eh) in eyes3:
        #     ex, ey, ew, eh = to_src_coord(ex, ey, ew, eh)
        #     cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        #
        # # both eyes simple
        # # yellow
        # eyes4 = self.eye_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        # for (ex, ey, ew, eh) in eyes4:
        #     ex, ey, ew, eh = to_src_coord(ex, ey, ew, eh)
        #     cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (66, 244, 244), 1)

        # left eye
        # red
        eyes = self.lefteye_2splits_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (x, y, w, h) in eyes:
            l_eye_img = face_img[y:y + h, x:x + w]
            l_eye_rect = (x, y, w, h)
            x, y, w, h = to_src_coord(x, y, w, h)
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # right eye
        # pink
        eyes2 = self.righteye_2splits_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (x, y, w, h) in eyes2:
            r_eye_img = face_img[y:y + h, x:x + w]
            x, y, w, h = to_src_coord(x, y, w, h)
            r_eye_rect = (x, y, w, h)
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (137, 66, 244), 1)

        return l_eye_img, r_eye_img, l_eye_rect, r_eye_rect
