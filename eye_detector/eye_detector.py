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

    face_width = 0
    face_height = 0

    cv2.ocl.setUseOpenCL(False)

    try_to_find_second_time = 0
    found_second_time = 0

    # на вход подается лицо (квадрат) в серых тонах
    def get_eyes(self, src_img, face_img, face_rect):

        height, width = face_img.shape
        self.face_width = width
        self.face_height = height

        # ищем глаза только в верхней части лица
        # face_img = face_img[0:int(height * 0.6), 0:width]
        face_img = face_img[0:int(height * 1), 0:width]

        # self.min_eye_size = (int(width / 10), int(width / 10))
        # self.max_eye_size = (int(width / 3), int(width / 3))

        self.min_eye_size = (int(width / 100), int(width / 100))
        self.max_eye_size = (int(width / 1), int(width / 1))

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
        use_complement0 = False
        use_complement1 = False

        if len(possible_l_eyes) == 0 or len(possible_r_eyes) == 0:
            use_complement0 = True
            self.complement_eye_variants(face_img, self.eye_cascade, possible_l_eyes, possible_r_eyes,
                                        can_be_left, can_be_right)

        if len(possible_l_eyes) == 0 or len(possible_r_eyes) == 0:
            use_complement1 = True
            self.complement_eye_variants(face_img, self.eyeglasses_cascade, possible_l_eyes, possible_r_eyes,
                                        can_be_left, can_be_right)

        l_eye, r_eye = self.choose_best_eyes(possible_l_eyes, possible_r_eyes)
        if l_eye is None or r_eye is None:
            # если хотя бы одного из глаз не хватает, пытаемся доискать

            if not use_complement0:
                self.complement_eye_variants(face_img, self.eye_cascade, possible_l_eyes, possible_r_eyes,
                                             can_be_left, can_be_right)
            if not use_complement1:
                self.complement_eye_variants(face_img, self.eyeglasses_cascade, possible_l_eyes, possible_r_eyes,
                                             can_be_left, can_be_right)

            if not use_complement0 or not use_complement1:
                l_eye, r_eye = self.choose_best_eyes(possible_l_eyes, possible_r_eyes)

                self.try_to_find_second_time += 1
                if l_eye is not None and r_eye is not None:
                    self.found_second_time += 1


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


    # статистика по дополнению результатов
    complement_eye_variants_stat = 0

    # дополняет результаты поиска глаза, если он еще не найден
    def complement_eye_variants(self, img, haar, possible_l_eyes, possible_r_eyes, can_be_left, can_be_right):
        possible_eyes = []

        self.add_eye_variants(img, haar,
                              self.scale_factor, self.min_neighbors, lambda x, y, w, h: True, possible_eyes)

        for eye_rect in possible_eyes:
            if len(possible_l_eyes) == 0 and can_be_left(*eye_rect):
                possible_l_eyes.append(eye_rect)
            elif len(possible_r_eyes) == 0 and can_be_right(*eye_rect):
                possible_r_eyes.append(eye_rect)
            if len(possible_l_eyes) != 0 and len(possible_r_eyes) != 0:
                self.complement_eye_variants_stat += 1
                break

    # статистика по случаю, когда все найденные пары глаз противоречивые
    all_eyes_pair_are_contradict = 0

    # выбирает конечный результат из всех вариантов пар
    def choose_best_eyes(self, possible_l_eyes, possible_r_eyes):

        res_l, res_r = self.choose_best_eye(possible_l_eyes), self.choose_best_eye(possible_r_eyes)

        if res_l is not None and res_r is not None and self.check_if_contradict(res_l, res_r):
            # глаза противоречат => не правильно выбрали
            # выбираем любую непротиворечивую пару

            for l_eye in possible_l_eyes:
                for r_eye in possible_r_eyes:
                    if not self.check_if_contradict(l_eye, r_eye):
                        return l_eye, r_eye

            # все пары противоречивые => возвращаем только один глаз
            self.all_eyes_pair_are_contradict += 1

            # выбираем правильный глаз
            l_rel = self.get_eye_pos_relevance(res_l, is_left=True)
            r_rel = self.get_eye_pos_relevance(res_l, is_left=False)
            return (res_l, None) if l_rel > r_rel else (None, res_r)

        return res_l, res_r


    def choose_best_eye(self, possible_eyes):
        # выбираем самый большой по размеру вариант
        max_size = 0
        res_rect = None
        for x, y, w, h in possible_eyes:
            if w * h > max_size:
                max_size = w * h
                res_rect = (x, y, w, h)

        return res_rect

    def check_if_intersects(self, rect0, rect1):
        x0, y0, w0, h0 = rect0
        x1, y1, w1, h1 = rect1

        if x0 > x1 + w1 or x1 > x0 + w0 :
            return False
        if y0 > y1 + h1 or y1 > y0 + h0:
            return False
        return True

    # возвращает True если положение двух глаз противоречит друг другу
    def check_if_contradict(self, rect0, rect1):
        x0, y0, w0, h0 = rect0
        x1, y1, w1, h1 = rect1

        c0_x = x0 + w0 / 2
        c0_y = y0 + h0 / 2
        c1_x = x1 + w1 / 2
        c1_y = y1 + h1 / 2

        # проверяем, что середины глаз не лежат внутри другого глаза
        # (сами прямоугольники могут пресекаться)

        if self.check_if_intersects(rect0, (c1_x, c1_y, 0, 0)):
            return True
        if self.check_if_intersects(rect1, (c0_x, c0_y, 0, 0)):
            return True
        return False

    def get_eye_pos_relevance(self, rect, is_left):
        target_x = int(self.face_width * 0.35) if is_left else int(self.face_width * 0.65)
        x, y, w, h = rect
        return abs(x + w / 2 - target_x);



