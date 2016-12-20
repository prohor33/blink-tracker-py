import cv2
import numpy as np
from data_manager import utils
import os

class EyeStateDetector:

    img_def_size = 24

    def load_train_data(self, src_dir):
        for filename in os.listdir(src_dir):
            # предполагается, что картинки уже в серых тонах
            img = cv2.imread(src_dir + '/' + filename)
            if img is None:
                print('error: no image')
                continue

            h, w = utils.get_img_size(img)
            if h != w:
                print('error: img not a square')
                continue

            if h != self.img_def_size:
                img = utils.resize_img_to(img, self.img_def_size)

            data = img[:, :, 0]
            print(data.shape)
            data = np.array(data)
            data = data.flatten()
            print(data.shape)
            print(data)

            # cv2.imshow('data', data)

            break



    # eye_img - 24x24 картинка глаза в серых тонах
    # возвращает вероятность того, что глаз закрыт от 0 до 100
    # def get_eye_state(self, src_img, eye_img, eye_rect):

