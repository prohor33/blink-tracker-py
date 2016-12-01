import cv2
import numpy as np
import math

def one_img_from_two(img1, img2):
    return np.concatenate((img1, img2), axis=1)

def one_img_from_three(img1, img2, img3):
    return np.concatenate((img1, img2, img3), axis=1)

def distance(p0, p1):
    return math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2))

def get_box_area(b):
    return distance(b[0], b[1]) * distance(b[1], b[2])

def scale_points(points, factor):
    for p in points:
        p[0] = p[0] * factor
        p[1] = p[1] * factor

def scale_cnt(cnt, factor):
    for points in cnt:
        scale_points(points, factor)

def scale_countours(contours, factor):
    for cnt in contours:
        scale_cnt(cnt, factor)

def scale_img(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)