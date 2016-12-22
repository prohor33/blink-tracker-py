#!/usr/bin/env python3

from data_manager import data_manager

data_mgr = data_manager.DataManager()

# data_mgr.run_face_detection('data/faces_original')
# data_mgr.run_eye_detection('data/faces_found')
# data_mgr.simple_detection('data/eyes_found_tmp')
# data_mgr.run_eye_shape_detection('data/eyes_found_big')
data_mgr.run_on_video('data/vanya/2.mov')
# data_mgr.run_eye_state_detection('data/train_dataset')