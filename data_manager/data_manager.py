#!/usr/bin/env python3
from face_detector import face_detector
from blink_detector import blink_detector
import os
import cv2
import shutil


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

    def run_on_video(self, filename):
        cap = cv2.VideoCapture(filename)

        blink_det = blink_detector.BlinkDetector()

        while (True):
            cap.grab()

            # Capture frame-by-frame
            ret, frame = cap.retrieve()

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