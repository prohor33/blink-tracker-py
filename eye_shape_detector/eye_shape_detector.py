import cv2
from itertools import cycle
import numpy as np
from data_manager import utils

class EyeShapeDetector:

    # на вход подается квадрат: глаз и то, что вокруг него в серых тонах
    def get_shape(self, src_img):
        h, w, channels = src_img.shape

        thrs_percent = 8
        threshold_res, threshold_vis = utils.threshold_up_to_percent(src_img, thrs_percent)

        # находим контуры
        _, contours0, hierarchy = cv2.findContours(threshold_res.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]


        # edge_linking_thr = 0
        # initial_segment_strongedges = 1850
        edge_linking_thr = 754
        initial_segment_strongedges = 1460
        canny_img = cv2.Canny(src_img, threshold1=edge_linking_thr, threshold2=initial_segment_strongedges, apertureSize=5)

        res_canny = src_img.copy()
        crosses_img = src_img.copy()

        res_canny[canny_img != 0] = (0, 255, 0)

        # крестик где предполагаем центр глаза
        cross_p = [int(w / 2), int(0.65 * h)]
        cross_s = [int(w * 0.9), int(h * 0.45)]

        levels = 1

        # находим прямоугольники
        # res_img3 = np.zeros((h, w, 3), np.uint8)
        boxes = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)

        area_coef = 0.0
        boxes.sort(key=lambda box: -utils.distance(utils.get_box_center(box), cross_p) + utils.get_box_area(box) * area_coef)
        ok_boxes = boxes[-1:]

        scale_factor = 5
        for box in boxes:
            utils.scale_points(box, scale_factor)
        utils.scale_countours(contours, scale_factor)
        utils.scale_points([cross_p], scale_factor)
        utils.scale_points([cross_s], scale_factor)

        src_img = utils.scale_img(src_img, scale_factor)
        crosses_img = utils.scale_img(crosses_img, scale_factor)
        threshold_vis = utils.scale_img(threshold_vis, scale_factor)
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

        for box in ok_boxes:
            cv2.drawContours(res_img4, [box], 0, (0, 0, 255), line_w)

        utils.draw_box(crosses_img, cross_p, cross_s, color=(0, 0, 255))
        utils.draw_box(crosses_img, utils.get_box_center(box), cross_s, color=(0, 255, 0))

        return src_img, threshold_vis, res_img2, res_img3, res_img4, crosses_img

