import numpy as np
from .detector import Detector
import cv2
from skimage.feature import hog


class HOG(Detector):
    def __init__(self, threshold = 60) -> None:
        self.threshold = threshold


    def get_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
        return hog_image


    def draw_keypoints(self, img):
        gray = self.get_keypoints(img)
        result = np.copy(img)

        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                if gray[y, x] > self.threshold:
                    result = cv2.circle(result, (x, y), 1 , (0, 255, 0), 1)
        return result