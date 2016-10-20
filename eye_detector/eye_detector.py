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
    def get_eyes(self, src_img, face_img, face_p):

        left_eye_img = None
        right_eye_img = None

        face_x = face_p[0]
        face_y = face_p[1]

        # left eye
        # red
        eyes = self.lefteye_2splits_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (ex, ey, ew, eh) in eyes:
            ex += face_x
            ey += face_y
            cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # right eye
        # pink
        eyes2 = self.righteye_2splits_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (ex, ey, ew, eh) in eyes2:
            ex += face_x
            ey += face_y
            cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (137, 66, 244), 2)

        # both eyes
        # green
        eyes3 = self.eyeglasses_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (ex, ey, ew, eh) in eyes3:
            ex += face_x
            ey += face_y
            cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # both eyes simple
        # yellow
        eyes4 = self.eye_cascade.detectMultiScale(face_img, self.scale_factor, self.min_neighbors)
        for (ex, ey, ew, eh) in eyes4:
            ex += face_x
            ey += face_y
            cv2.rectangle(src_img, (ex, ey), (ex + ew, ey + eh), (66, 244, 244), 2)

        return left_eye_img, right_eye_img
