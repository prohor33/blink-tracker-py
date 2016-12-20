#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

# built-in module
import sys


if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    cv2.namedWindow('res')
    cv2.namedWindow('res1')
    cv2.namedWindow('res2')
    cv2.moveWindow('res', 250, 100)
    cv2.moveWindow('res1', 0, 300)
    cv2.moveWindow('res2', 250, 300)
    # cv2.createTrackbar('thrs1', 'edge', 1, 20, nothing)
    cv2.createTrackbar('thrs1', 'edge', 0, 250, nothing)
    # cv2.createTrackbar('thrs1', 'edge', 0, 5000, nothing)
    # cv2.createTrackbar('thrs1', 'edge', 3, 7, nothing)
    # cv2.createTrackbar('thrs2', 'edge', 1160, 5000, nothing)
    # cv2.createTrackbar('thrs2', 'edge', 0, 5000, nothing)

    img = cv2.imread('../data/eyes_found_tmp/6.jpg')
    if img is None:
        print('warning: no image')
    # img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    src_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    while True:
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        # print(thrs1, thrs2)
        res = cv2.Canny(img, thrs1, thrs2, apertureSize=5)
        retv, edge = cv2.threshold(img, thresh=thrs1, maxval=2,  type=cv2.THRESH_TOZERO)
        # res_x = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
        # res_y = cv2.Sobel(img, ddepth=-1, dx=0, dy=1)
        # res = cv2.addWeighted(res_x, 1.0, res_y, 1.0, 0)

        vis = img.copy()
        # vis = np.uint8(vis/2.)
        vis[res != 0] = (0, 255, 0)

        # print('thrs1 = ', thrs1, ' thrs2 = ', thrs2)

        h, w = res.shape[:2]
        _, contours0, hierarchy = cv2.findContours(res.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
        vis2 = np.zeros((h, w, 3), np.uint8)
        levels = thrs1 - 3
        cv2.drawContours(vis2, contours, (-1, 2)[levels <= 0], (128, 255, 255),
                         3, cv2.LINE_AA, hierarchy, abs(levels))

        vis3 = np.zeros((h, w, 3), np.uint8)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(vis3, [box], 0, (0, 0, 255), 2)

        vis = cv2.resize(vis, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        res = cv2.resize(res, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        vis2 = cv2.resize(vis2, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        vis3 = cv2.resize(vis3, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        img_big = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        edge = cv2.resize(edge, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        # res_x = cv2.resize(res_x, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        # res_y = cv2.resize(res_y, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('edge', edge)
        cv2.imshow('res', res)
        # cv2.imshow('res1', src_img)
        cv2.imshow('res1', img_big)
        # cv2.imshow('edge', vis2)


        # cv2.imshow('res2', res_y)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
    cv2.destroyAllWindows()