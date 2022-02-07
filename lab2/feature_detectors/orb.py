import cv2
import numpy as np
from .detector import Detector


class ORB(Detector):
    def __init__(self) -> None:
        self.orb = cv2.ORB_create()

    
    def get_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp


    def draw_keypoints(self, img):
        kp = self.get_keypoints(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_copy = np.copy(img)
        image_copy = cv2.drawKeypoints(image_copy, kp, image_copy)
        return image_copy