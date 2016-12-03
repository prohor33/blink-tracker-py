import cv2
from itertools import cycle
import numpy as np
from data_manager import utils

class EyeShapeDetector:

    # на вход подается квадрат: глаз и то, что вокруг него в серых тонах
    def get_shape(self, src_img):
        thrs_percent = 20
        threshold_res = utils.threshold_up_to_percent(src_img, thrs_percent)

        # edge_linking_thr = 0
        # initial_segment_strongedges = 1850
        edge_linking_thr = 754
        initial_segment_strongedges = 1460
        canny_img = cv2.Canny(src_img, threshold1=edge_linking_thr, threshold2=initial_segment_strongedges, apertureSize=5)

        res_img0 = src_img.copy()
        res_img1 = src_img.copy()

        # vis = img.copy()
        # vis = np.uint8(vis / 2.)
        res_img0[canny_img != 0] = (0, 255, 0)

        str_modes = ['ellipse', 'rect', 'cross']
        cur_str_mode = str_modes[0]

        # не особо нужная морфология
        sz = 2
        st = cv2.getStructuringElement(getattr(cv2, 'MORPH_' + cur_str_mode.upper()), (sz, sz))
        res2 = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, st)
        res_img1[res2 != 0] = (0, 255, 0)

        h, w, channels = src_img.shape

        # рисуем крестик
        st_p = (int(w / 2), int(0.65 * h))
        cross_s = (int(w * 0.9), int(h * 0.45))
        cv2.line(res_img1, (int(st_p[0] - cross_s[0]), st_p[1]), (int(st_p[0] + cross_s[0]), st_p[1]), color=(0, 0, 255))
        cv2.line(res_img1, (st_p[0], int(st_p[1] - cross_s[1])), (st_p[0], int(st_p[1] + cross_s[1])),
                 color=(0, 0, 255))

        # находим контуры
        _, contours0, hierarchy = cv2.findContours(canny_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
        # res_img2 = np.zeros((h, w, 3), np.uint8)
        levels = 1

        # находим прямоугольники
        # res_img3 = np.zeros((h, w, 3), np.uint8)
        boxes = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)

        boxes.sort(key=lambda box: utils.get_box_area(box))
        ok_boxes = boxes[-2:]

        scale_factor = 5
        for box in boxes:
            utils.scale_points(box, scale_factor)
        utils.scale_countours(contours, scale_factor)

        src_img = utils.scale_img(src_img, scale_factor)
        res_img0 = utils.scale_img(res_img0, scale_factor)
        threshold_res = utils.scale_img(threshold_res, scale_factor)
        res_img2 = src_img.copy()
        res_img3 = src_img.copy()
        res_img4 = src_img.copy()
        h = h * scale_factor
        w = w * scale_factor

        line_w = int(scale_factor / 3)
        cv2.drawContours(res_img2, contours, (-1, 2)[levels <= 0], (128, 255, 255),
                         line_w, cv2.LINE_AA, hierarchy, abs(levels))

        for box in boxes:
            cv2.drawContours(res_img3, [box], 0, (0, 0, 255), line_w)

        # res_img4 = np.zeros((h, w, 3), np.uint8)
        for box in ok_boxes:
            cv2.drawContours(res_img4, [box], 0, (0, 0, 255), line_w)

        return src_img, threshold_res, res_img0, res_img2, res_img3, res_img4

