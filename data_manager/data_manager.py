#!/usr/bin/env python3
from face_detector import face_detector
import os
import cv2
import shutil


class DataManager:

    def run_face_detection(self, src_dir):

        face_det = face_detector.FaceDetector();

        res_dir = src_dir + '/result/'
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        os.makedirs(res_dir)

        found = 0
        not_found = 0

        for filename in os.listdir(src_dir):
            img = cv2.imread(src_dir + '/' + filename)
            if img is None:
                print('error: no image')
                continue
            res_img = face_det.get_face(img)
            if res_img is None:
                print('no face found')
                not_found += 1
            else:
                found += 1
                size = 100
                res_img = cv2.resize(res_img, (size, size), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(res_dir + filename, res_img)


        print('found: ' + str(found))
        print('not found: ' + str(not_found))