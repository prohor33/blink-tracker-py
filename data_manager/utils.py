import cv2
import numpy as np

def one_img_from_two(img1, img2):
    return np.concatenate((img1, img2), axis=1)

def one_img_from_three(img1, img2, img3):
    return np.concatenate((img1, img2, img3), axis=1)