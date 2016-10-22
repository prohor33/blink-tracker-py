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

    min_eye_size = (0, 0)
    max_eye_size = (0, 0)

    cv2.ocl.setUseOpenCL(False)

    # на вход подается лицо (квадрат) в серых тонах
    def get_eyes(self, src_img, face_img, face_rect):

        height, width = face_img.shape

        # ищем глаза только в верхней части лица
        face_img = face_img[0:int(height * 0.6), 0:width]

        self.min_eye_size = (int(width / 10), int(width / 10))
        self.max_eye_size = (int(width / 3), int(width / 3))

        possible_l_eyes = []
        possible_r_eyes = []

        can_be_left = lambda x, y, w, h: x + w > width / 2
        can_be_right = lambda x, y, w, h: x < width / 2

        # левый глаз
        self.add_eye_variants(face_img, self.lefteye_2splits_cascade,
                         self.scale_factor, self.min_neighbors, can_be_left, possible_l_eyes)

        # правый глаз
        self.add_eye_variants(face_img, self.righteye_2splits_cascade,
                         self.scale_factor, self.min_neighbors, can_be_right, possible_r_eyes)

        # делаем дополнение, если не нашли
        self.complement_eye_variants(face_img, self.eye_cascade, possible_l_eyes, possible_r_eyes,
                                     can_be_left, can_be_right)
        self.complement_eye_variants(face_img, self.eyeglasses_cascade, possible_l_eyes, possible_r_eyes,
                                     can_be_left, can_be_right)

        l_eye, r_eye = self.choose_best_eyes(possible_l_eyes, possible_r_eyes)

        # возвращаем результат

        # лицо в изначальной картинке
        face_x = face_rect[0][0]
        face_y = face_rect[0][1]
        face_w = face_rect[1][0]
        face_h = face_rect[1][1]

        to_src_coord = lambda x, y, w, h: (int(face_x + x / width * face_w), int(face_y + y / height * face_h),
                                           int(w / width * face_w), int(h / height * face_h))

        l_eye_img = None
        r_eye_img = None

        if l_eye is not None:
            x, y, w, h = l_eye
            l_eye_img = face_img[y:y + h, x:x + w]
            x, y, w, h = to_src_coord(*l_eye)
            # левый розовый
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (137, 66, 244), 1)  # BGR

        if r_eye is not None:
            x, y, w, h = r_eye
            r_eye_img = face_img[y:y + h, x:x + w]
            x, y, w, h = to_src_coord(*r_eye)
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return l_eye_img, r_eye_img, l_eye, r_eye


    # запускает каскад и добавляет результаты
    def add_eye_variants(self, img, haar, scale_factor, min_neighbors, can_be_eye, possible_eyes):
        eyes = haar.detectMultiScale(img, scale_factor, min_neighbors, 0, self.min_eye_size, self.max_eye_size)
        for (x, y, w, h) in eyes:
            if not can_be_eye(x, y, w, h):
                continue
            possible_eyes.append((x, y, w, h))

    # дополняет результаты поиска глаза, если он еще не найден
    def complement_eye_variants(self, img, haar, possible_l_eyes, possible_r_eyes, can_be_left, can_be_right):
        if len(possible_l_eyes) == 0 or len(possible_r_eyes) == 0:
            possible_eyes = []

            self.add_eye_variants(img, haar,
                                  self.scale_factor, self.min_neighbors, lambda x, y, w, h: True, possible_eyes)

            for eye_rect in possible_eyes:
                if len(possible_l_eyes) == 0 and can_be_left(*eye_rect):
                    possible_l_eyes.append(eye_rect)
                elif len(possible_r_eyes) == 0 and can_be_right(*eye_rect):
                    possible_r_eyes.append(eye_rect)
                if len(possible_l_eyes) != 0 and len(possible_r_eyes) != 0:
                    break


    def choose_best_eyes(self, possible_l_eyes, possible_r_eyes):
        return self.choose_best_eye(possible_l_eyes), self.choose_best_eye(possible_r_eyes)

    def choose_best_eye(self, possible_eyes):
        # выбираем самый большой по размеру вариант
        max_size = 0
        res_rect = None
        for x, y, w, h in possible_eyes:
            if w * h > max_size:
                max_size = w * h
                res_rect = (x, y, w, h)

        return res_rect