import cv2
import numpy as np
from .detector import Detector

class HarrisCornerDetector(Detector):

    def __init__(self, threshold=0.1) -> None:
        self.threshold = threshold


    def get_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dist = cv2.cornerHarris(gray, 2, 3, 0.04)
        dist = cv2.dilate(dist, None)
        return dist
    

    def draw_keypoints(self, img):
        gray_dist = self.get_keypoints(img)
        thresh = self.threshold * gray_dist.max()
        result = np.copy(img)
        for y in range(gray_dist.shape[0]):
            for x in range(gray_dist.shape[1]):
                if gray_dist[y, x] > thresh:
                    result = cv2.circle(result, (x, y), 1 , (0, 255, 0), 1)
        return result
