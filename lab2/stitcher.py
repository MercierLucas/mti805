import cv2


class Stitcher:

    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher_create()


    def generate_panorama(self, images):
        (status, stitched) = self.stitcher.stitch(images)
        if status == 0:
            return stitched
        return None