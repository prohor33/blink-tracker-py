import cv2

class EyeShapeDetector:

    # на вход подается квадрат: глаз и то, что вокруг него в серых тонах
    def get_shape(self, src_img):
        res = cv2.Canny(src_img, threshold1=0, threshold2=1850, apertureSize=5)

        res_img0 = src_img.copy()
        res_img1 = src_img.copy()

        # vis = img.copy()
        # vis = np.uint8(vis / 2.)
        res_img0[res != 0] = (0, 255, 0)

        sz = 2
        st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        res2 = cv2.morphologyEx(res, cv2.MORPH_CLOSE, st)
        res_img1[res2 != 0] = (0, 255, 0)

        return res_img0, res_img1

