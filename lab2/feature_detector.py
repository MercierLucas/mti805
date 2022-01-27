import cv2

"""
Mainly from:
    https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
"""
class FeatureDetector:
    methods = {
        'sift': cv2.SIFT_create,
        'orb': cv2.ORB_create
    }

    def __init__(self, method = 'sift') -> None:
        assert method in self.methods, 'Unknown method'
        self.detector = self.methods[method]()

    
    def draw_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, _ = self.get_keypoints(image)
        image_kp = cv2.drawKeypoints(gray, kp, image)
        return image_kp

    
    def get_keypoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    
    
    def match(self, img1, img2, top_k=10):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        kp1, des1 = self.get_keypoints(img1)
        kp2, des2 = self.get_keypoints(img2)

        matches = matcher.match(des1,des2)

        matches = sorted(matches, key = lambda x:x.distance)
        img_match = cv2.drawMatches(img1,kp1,img2,kp2,matches[:top_k],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_match

