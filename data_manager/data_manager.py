#!/usr/bin/env python3
from face_detector import face_detector
from eye_detector import eye_detector
from blink_detector import blink_detector
from eye_shape_detector import eye_shape_detector
import os
import cv2
import shutil
from .utils import *


class DataManager:

    def run_face_detection(self, src_dir):

        face_det = face_detector.FaceDetector()

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
            face_rect, face_img = face_det.get_face(img)
            if face_img is None:
                print('no face found')
                not_found += 1
            else:
                found += 1
                size = 100
                face_img = cv2.resize(face_img, (size, size), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(res_dir + filename, face_img)


        print('found: ' + str(found))
        print('not found: ' + str(not_found))


    def run_eye_detection(self, src_dir):

        eye_det = eye_detector.EyeDetector()

        eyes_position = {}

        # считываем файл с результатами
        answer_file = open(src_dir + '/../EyeCoordinatesInfo_OpenFace.txt', 'r')
        for line in answer_file:
            tmp_str = line.split()
            filename = tmp_str[0]
            x0 = int(tmp_str[1])
            y0 = int(tmp_str[2])
            x1 = int(tmp_str[3])
            y1 = int(tmp_str[4])
            eyes_position[filename] = (x0, y0, x1, y1)



        res_dir = src_dir + '_res/'
        l_res_dir = src_dir + '_res/left_eye/'
        r_res_dir = src_dir + '_res/right_eye/'
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        os.makedirs(l_res_dir)
        os.makedirs(r_res_dir)

        found_both = 0
        found_one = 0
        not_found = 0

        for filename in os.listdir(src_dir):
            img = cv2.imread(src_dir + '/' + filename)
            if img is None:
                print('warning: no image')
                continue

            height, width, channels = img.shape
            face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            l_eye_img, r_eye_img, l_eye_rect, r_eye_rect = eye_det.get_eyes(img, face_img, ((0, 0), (width, height)))

            eyes_pos = eyes_position[filename]

            is_eye_ok = lambda x_or, y_or, x, y, w, h: x_or < x or x_or > (x + w) or y_or < y or y_or > (y + h)

            l_eye_is_found = False
            r_eye_is_found = False

            size = 24
            if l_eye_img is not  None:

                l_eye_is_found = is_eye_ok(eyes_pos[0], eyes_pos[1],
                                           l_eye_rect[0], l_eye_rect[1], l_eye_rect[2], l_eye_rect[3])

                l_eye_img = cv2.resize(l_eye_img, (size, size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(l_res_dir + filename, l_eye_img)

            if r_eye_img is not None:
                r_eye_is_found = is_eye_ok(eyes_pos[2], eyes_pos[3],
                                           r_eye_rect[0], r_eye_rect[1], r_eye_rect[2], r_eye_rect[3])

                r_eye_img = cv2.resize(r_eye_img, (size, size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(r_res_dir + filename, r_eye_img)

            if l_eye_is_found and r_eye_is_found:
                found_both += 1
            else:
                if l_eye_is_found or r_eye_is_found:
                    found_one += 1
                else:
                    not_found += 1

            # cv2.imwrite(res_dir + filename, img)

        print('found both: ' + str(found_both))
        print('found one: ' + str(found_one))
        print('not found: ' + str(not_found))
        print('complement eye variants: ' + str(eye_det.complement_eye_variants_stat))
        print('all eyes pair are contradict: ' + str(eye_det.all_eyes_pair_are_contradict))

    def run_on_video(self, filename):
        cap = cv2.VideoCapture(filename)

        blink_det = blink_detector.BlinkDetector()

        while (True):
            cap.grab()

            # Capture frame-by-frame
            ret, frame = cap.retrieve()

            height, width, chanel = frame.shape

            # уменьшаем размер
            max_size = 200
            transform_factor = max_size / max(width, height)
            if transform_factor < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=transform_factor, fy=transform_factor)

            if not ret:
                print("error: no frame")
                break

            blink_det.detect(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                # time.sleep(0.2)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


    eye_cascade = cv2.CascadeClassifier('data/haars/haarcascade_eye.xml')

    def simple_detection(self, src_dir):

        for filename in os.listdir(src_dir):
            file_path = src_dir + '/' + filename
            img = cv2.imread(file_path)
            if img is None:
                print('warning: no image')
                continue

            eyes = self.eye_cascade.detectMultiScale(img, 1.01, 1)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imwrite(file_path, img)


    def run_eye_shape_detection(self, src_dir):
        res_dir = src_dir + '_eye_shape/'
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        os.makedirs(res_dir)

        eye_shape_det = eye_shape_detector.EyeShapeDetector()

        for filename in os.listdir(src_dir):
            img = cv2.imread(src_dir + '/' + filename)
            if img is None:
                print('warning: no image')
                continue
            src_img = img.copy()

            results = eye_shape_det.get_shape(img)

            res_img = np.concatenate(results, axis=1)
            # res_img = res1

            # size_coef = 4
            # size_coef = 3
            # res_img = cv2.resize(res_img, None, fx=size_coef, fy=size_coef, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(res_dir + filename, res_img)
