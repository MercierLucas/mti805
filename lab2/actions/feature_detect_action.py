from .action import Action
from feature_detectors import FeatureDetectorManager



class FeatureDetectAction(Action):
    def __init__(self, args, required_input_type=None) -> None:
        super().__init__(args, required_input_type)
        detector = args['feature_detector']
        if detector == 'all':
            self.detector = None
        else:
            self.detector = FeatureDetectorManager(detector=detector)

    
    def perform(self):
        if self.detector is None:
            images_with_kp = []
            titles = []
            for detector_name in FeatureDetectorManager.detectors:
                titles.append(detector_name)
                detector = FeatureDetectorManager(detector=detector_name)
                images_with_kp.append(detector.draw_keypoints(self.input[0]))
            self.show_result(images_with_kp, titles)
            return

        match_image = self.detector.draw_keypoints(self.input[0])
        self.show_result(match_image)
        