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

def get_middle(p0, p1):
    return [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]

def get_box_center(b):
    return get_middle(b[0], b[2])

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

def get_percent_of_threshold(img, thrs):
    h, w = img.shape
    retv, threshold_res = cv2.threshold(img, thresh=thrs, maxval=1,  type=cv2.THRESH_TOZERO)
    return [1.0 - cv2.countNonZero(threshold_res) / (h * w), threshold_res]

stat_threshold_res = {}

def threshold_up_to_percent(src_img, target_percent):
    h, w = src_img.shape
    target_percent = target_percent / 100.0
    img = src_img[:, :]
    result = []
    thrsholds = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
    for thrs in thrsholds:
        res = [thrs]
        res.extend(get_percent_of_threshold(img, thrs))
        result.append(res)
    result = sorted(result, key=lambda x: abs(x[1] - target_percent))
    res_img = result[0][2]
    thrs_val = result[0][0]
    # stat_threshold_res[thrs_val] = stat_threshold_res.get(thrs_val, 0) + 1
    # print(stat_threshold_res)
    threshold_res = np.zeros((h, w, 1), np.uint8)
    threshold_res[res_img == 0] = 255
    src_img2 = src_img.copy()
    src_img2[res_img == 0] = 255
    return threshold_res, src_img2

def draw_box(img, c, size, color, thickness=1):
    s = size[:]
    s[0] = int(s[0] / 2.0)
    s[1] = int(s[1] / 2.0)
    p0 = (int(c[0] - s[0]), int(c[1] - s[1]))
    p1 = (int(c[0] + s[0]), int(c[1] - s[1]))
    p2 = (int(c[0] + s[0]), int(c[1] + s[1]))
    p3 = (int(c[0] - s[0]), int(c[1] + s[1]))

    cv2.line(img, p0, p1, color=color, thickness=thickness)
    cv2.line(img, p1, p2, color=color, thickness=thickness)
    cv2.line(img, p2, p3, color=color, thickness=thickness)
    cv2.line(img, p3, p0, color=color, thickness=thickness)

def draw_rect(img, rect, color, thickness=1):
    c, s = rect_to_box(rect)
    draw_box(img, c, s, color, thickness)

# box center + size
# rect top left, size
def adjust_brightness(img, brightness):
    h, w = img.shape
    val = 0.0
    for x in range(0, w):
        for y in range(0, h):
            val = val + img[y][x]
    val = val / (h * w)

    for x in range(0, w):
        for y in range(0, h):
            img[y][x] = np.clip(img[y][x] / val * brightness, 0, 255)

def convert_rect_to_parent(img, rect_in_parent, rect):
    height, width = img.shape
    x = rect[0][0]
    y = rect[0][1]
    w = rect[1][0]
    h = rect[1][1]
    x_in_p = rect_in_parent[0][0]
    y_in_p = rect_in_parent[0][1]
    w_in_p = rect_in_parent[1][0]
    h_in_p = rect_in_parent[1][1]
    return [[int(x_in_p + x / width * w_in_p), int(y_in_p + y / height * h_in_p)],
                                           [int(w / width * w_in_p), int(h / height * h_in_p)]]

def box_to_rect(c, s):
    return [[c[0] - s[0] / 2.0, c[1] - s[1] / 2.0], [s[0], s[1]]]

def get_rect_center(r):
    return [r[0][0] + r[1][0] / 2.0, r[0][1] + r[1][1] / 2.0]

def rect_to_box(r):
    c = get_rect_center(r)
    s = r[1]
    return c, s

def crop_img_by_rect(img, r):
    return img[r[0][1] : r[0][1] + r[1][1], r[0][0] : r[0][0] + r[1][0]]

def tuple_to_rect(t):
    return [[t[0], t[1]], [t[2], t[3]]]

def get_img_size(img):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    return h, w

def resize_img_to(img, max_size):
    h, w = get_img_size(img)
    transform_factor = max_size / max(w, h)
    return cv2.resize(img, (0, 0), fx=transform_factor, fy=transform_factor)