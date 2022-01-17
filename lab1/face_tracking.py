import numpy as np
from face_detector import FaceDetector
from faceid import FaceID
from utils.viz_utils import add_rectangles

class FaceTracking:
    """Contains both face detector and faceid"""

    def __init__(self, face_detector:FaceDetector, face_id:FaceID) -> None:
        self.face_detector = face_detector
        self.faceid = face_id
        self.colors = {}


    def _get_matching_color(self, label):
        if label in self.colors:
            return self.colors[label]
        self.colors[label] = list(np.random.random(size=3) * 256)
        return self.colors[label]
    

    def track(self, image, verbose=True):
        faces, positions = self.face_detector.face_detection(image, return_pos=True)
        if len(faces) == 0:
            return image

        for face, pos in zip(faces, positions):
            prob, label_pred = self.faceid.recognize(face)
            if verbose:
                print(f'Pred: {label_pred} {prob}')
            prob *= 100
            image = add_rectangles(image, [pos], label=f'{label_pred}: {prob:.0f}%', color=self._get_matching_color(label_pred))
        return image


    