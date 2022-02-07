import cv2
from .harris import HarrisCornerDetector
from .sift import SIFT
from .orb import ORB
from .hog import HOG

"""
Mainly from:
    https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
"""

class FeatureDetectorManager:
    detectors = {
        'harris': HarrisCornerDetector,
        'sift': SIFT,
        'orb': ORB,
        'hog': HOG
    }

    def __init__(self, detector = 'harris') -> None:
        assert detector in self.detectors, 'Unknown method'
        self.detector = self.detectors[detector]()

    
    def draw_keypoints(self, image):
        image_kp = self.detector.draw_keypoints(image)
        return image_kp

    
    def get_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    

