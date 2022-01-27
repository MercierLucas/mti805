from .action import Action
from feature_detector import FeatureDetector



class MatchAction(Action):
    def __init__(self, args, required_input_type=None) -> None:
        super().__init__(args, required_input_type)

    
    def perform(self):
        detector = FeatureDetector(method='orb')
        match_image = detector.match(self.input[0], self.input[1], top_k=50)
        self.show_result(match_image)