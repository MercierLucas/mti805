class Detector:

    def __init__(self) -> None:
        pass

    def get_keypoints(self, img):
        raise NotImplementedError
    

    def draw_keypoints(self, img):
        raise NotImplementedError
