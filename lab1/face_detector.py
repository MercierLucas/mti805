import cv2
import numpy as np
from utils import add_rectangles


class FaceDetector:
    def __init__(self, cascade_root_path) -> None:
        self.face_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/face_cascade.xml')
        self.profile_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/profile_cascade.xml')
        self.eye_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/eye_cascade.xml')
        self.smile_cascade = cv2.CascadeClassifier(f'{cascade_root_path}/smile_cascade.xml')


    def _is_in_correct_portion(self, img, position, portion):
        height = img.shape[0]
        _, y, _, box_height = position
        y += box_height / 2
        return y <= height / 2 if portion == 'upper' else y >= height / 2
        if portion == 'upper':
            return y <= height / 2
        return y >= height / 2


    def _get_lowest(self, detections):
        if len(detections) == 0:
            return []
        max_detection = detections[0] # the y-axis goes from top to bottom, so max is lowest
        max_value = max_detection[1] # y
        for i, (_, y, _, _) in enumerate(detections):
            if y > max_value:
                max_detection = detections[i]
                max_value = y
        return [max_detection]


    def _detect_in_region(self, img, detector, start_x, start_y, max_items, portion, min_width_ratio=None):
        """Detect in specific region"""
        detected = detector.detectMultiScale(img, 1.3, 5)
        inside_pos = []
        for (x, y, w, h) in detected:
            if self._is_in_correct_portion(img, [x, y, w, h], portion):
                should_add = False
                if min_width_ratio is None:
                    should_add = True
                else:
                    ratio = w / img.shape[1]
                    should_add = ratio >= min_width_ratio
                if should_add:
                    inside_pos.append([x + start_x, y + start_y, w, h])

        if len(inside_pos) < max_items:
            return inside_pos

        return inside_pos[:max_items]


    def _stack_face_detections(self, faces, profiles):
        if len(faces) > 0 and len(profiles) > 0:
            return np.append(faces, profiles, 0)
        if len(faces) > 0 and len(profiles) == 0:
            return faces
        if len(profiles) > 0 and len(faces) == 0:
            return profiles
        return []


    def face_detection(self, img, return_pos):
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        profile = self.profile_cascade.detectMultiScale(img, 1.3, 5)
        #profile = []
        faces = self._stack_face_detections(faces, profile)

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
        profile = self.profile_cascade.detectMultiScale(img, 1.3, 5)
        faces = self._stack_face_detections(faces, profile)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            region = img[y: y + h, x: x + w]
            gray_region = gray[y: y + h, x: x + w]
            eyes = self._detect_in_region(gray_region, self.eye_cascade, x, y, max_items=10, portion='upper')
            smile = self._detect_in_region(region, self.smile_cascade, x, y, max_items=10, portion='lower', min_width_ratio=.4)
            smile = self._get_lowest(smile)
            img = add_rectangles(img, eyes, (0, 255, 0))
            img = add_rectangles(img, smile, (0, 0, 255))

        return img