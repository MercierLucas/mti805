import cv2
from utils import add_rectangles


class FaceDetector:
    def __init__(self, cascade_root_path) -> None:
        self.face_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/face_cascade.xml')
        self.eye_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/eye_cascade.xml')
        self.smile_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/smile_cascade.xml')


    def _detect_in_region(self, img, detector, start_x, start_y):
        """Detect in specific region"""
        detected = detector.detectMultiScale(img, 1.3, 5)
        inside_pos = [(x + start_x, y + start_y, w, h) for (x, y, w, h) in detected]
        return inside_pos


    def face_detection(self, img, return_pos):
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        regions = []
        for (x, y, w, h) in faces:
            regions.append(img[y:y+h, x:x+w])

        if return_pos:
            return regions, faces
        return regions


    def detect_and_add_shapes(self, img):
        # improvements: delete boxes that just appeared
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            region = img[y: y + h, x: x + w]
            gray_region = gray[y: y + h, x: x + w]
            eyes = self._detect_in_region(gray_region, self.eye_cascade, x, y)
            smile = self._detect_in_region(region, self.smile_cascade, x, y)
            img = add_rectangles(img, eyes, (0, 255, 0))
            img = add_rectangles(img, smile, (0, 0, 255))

        return img