from .action import Action
from stitcher import Stitcher



class StitchAction(Action):
    def __init__(self, args, required_input_type=None) -> None:
        super().__init__(args, required_input_type)

    
    def perform(self):
        stitcher = Stitcher()
        panorama = stitcher.generate_panorama(self.input)
        self.show_result(panorama)