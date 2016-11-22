import cv2
from itertools import cycle

class EyeShapeDetector:

    # на вход подается квадрат: глаз и то, что вокруг него в серых тонах
    def get_shape(self, src_img):
        res = cv2.Canny(src_img, threshold1=0, threshold2=1850, apertureSize=5)

        res_img0 = src_img.copy()
        res_img1 = src_img.copy()

        # vis = img.copy()
        # vis = np.uint8(vis / 2.)
        res_img0[res != 0] = (0, 255, 0)

        str_modes = ['ellipse', 'rect', 'cross']
        cur_str_mode = str_modes[0]

        # не особо нужная морфология
        sz = 2
        st = cv2.getStructuringElement(getattr(cv2, 'MORPH_' + cur_str_mode.upper()), (sz, sz))
        res2 = cv2.morphologyEx(res, cv2.MORPH_CLOSE, st)
        res_img1[res2 != 0] = (0, 255, 0)

        h, w, channels = src_img.shape

        # рисуем крестик
        st_p = (int(w / 2), int(0.65 * h))
        cross_s = (int(w * 0.9), int(h * 0.45))
        cv2.line(res_img1, (int(st_p[0] - cross_s[0]), st_p[1]), (int(st_p[0] + cross_s[0]), st_p[1]), color=(0, 0, 255))
        cv2.line(res_img1, (st_p[0], int(st_p[1] - cross_s[1])), (st_p[0], int(st_p[1] + cross_s[1])),
                 color=(0, 0, 255))

        # _, contours0, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return res_img0, res_img1

