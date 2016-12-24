import cv2
from itertools import cycle
import numpy as np
from data_manager import utils

class EyeShapeDetector:

    # на вход подается квадрат: глаз и то, что вокруг него в серых тонах
    def get_shape(self, src_img, eye_img, eye_rect):
        h, w = eye_img.shape

        thrs_percent = 8
        src_img_ths = eye_img.copy()
        utils.adjust_brightness(src_img_ths, 120)
        src_img_ths[0 : int(h/4.0)][:] = 255
        threshold_res, threshold_vis = utils.threshold_up_to_percent(src_img_ths, thrs_percent)

        # находим контуры
        _, contours0, hierarchy = cv2.findContours(threshold_res.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours0]


        # edge_linking_thr = 0
        # initial_segment_strongedges = 1850
        edge_linking_thr = 754
        initial_segment_strongedges = 1460
        canny_img = cv2.Canny(eye_img, threshold1=edge_linking_thr, threshold2=initial_segment_strongedges, apertureSize=5)

        res_canny = eye_img.copy()
        crosses_img = eye_img.copy()

        res_canny[canny_img != 0] = 255

        # крестик где предполагаем центр глаза
        cross_p = [int(w / 2), int(0.65 * h)]
        cross_s = [int(w * 0.9), int(h * 0.45)]
        res_size_coef = 0.9
        res_cross_s = [int(w * res_size_coef), int(h * res_size_coef)]

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

        # scale_factor = 5
        # for box in boxes:
        #     utils.scale_points(box, scale_factor)
        # utils.scale_countours(contours, scale_factor)
        # utils.scale_points([cross_p], scale_factor)
        # utils.scale_points([cross_s], scale_factor)
        # utils.scale_points([res_cross_s], scale_factor)
        #
        # eye_img = utils.scale_img(eye_img, scale_factor)
        # crosses_img = utils.scale_img(crosses_img, scale_factor)
        # threshold_vis = utils.scale_img(threshold_vis, scale_factor)
        # res_img2 = eye_img.copy()
        # res_img3 = eye_img.copy()
        # res_img4 = eye_img.copy()
        # res_img5 = eye_img.copy()
        # h = h * scale_factor
        # w = w * scale_factor

        # line_w = int(scale_factor / 3)
        # cv2.drawContours(res_img2, contours, (-1, 2)[levels <= 0], (128, 255, 255),
        #                  line_w, cv2.LINE_AA, hierarchy, abs(levels))
        #
        # for box in boxes:
        #     cv2.drawContours(res_img3, [box], 0, (0, 0, 255), line_w)
        #
        # for box in ok_boxes:
        #     cv2.drawContours(res_img4, [box], 0, (0, 0, 255), line_w)
        #
        # utils.draw_box(crosses_img, cross_p, cross_s, color=(0, 0, 255))

        res_box_center = cross_p
        # if len(ok_boxes) > 0:
        #     res_box_center = utils.get_box_center(ok_boxes[0])
        #     utils.draw_box(crosses_img, res_box_center, cross_s, color=(0, 255, 0))

        # оставили пока просто статическое положение
        # middle_box_center = utils.get_middle(res_box_center, cross_p)
        middle_box_center = cross_p

        # сдвинем результат внутрь области глаза
        # middle_box_center[0] = np.clip(middle_box_center[0], res_cross_s[0] / 2.0, w - res_cross_s[0] / 2.0)
        # middle_box_center[1] = np.clip(middle_box_center[1], res_cross_s[1] / 2.0, h - res_cross_s[1] / 2.0)

        # utils.draw_box(res_img5, middle_box_center, res_cross_s, color=(176, 29, 196), thickness=2)

        res_rect = utils.box_to_rect(middle_box_center, res_cross_s)
        res_rect = utils.convert_rect_to_parent(eye_img, eye_rect, res_rect)

        utils.draw_rect(src_img, res_rect, color=(176, 29, 196), thickness=1)

        return res_rect

        # return eye_img, threshold_vis, res_img2, res_img3, res_img4, crosses_img, res_img5

