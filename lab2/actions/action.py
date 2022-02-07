from os.path import isfile

from utils.loader import load_image, load_images
from utils.visualizer import show_image, show_images

class Action:

    INPUT_TYPE_FILE = 0
    INPUT_TYPE_FOLDER = 1
    INPUT_TYPE_WEBCAM = 2

    def __init__(self, args, required_input_type=None) -> None:
        self.args = args
        self.input_type = self.get_input_type(args['input'])

        if required_input_type:
            assert self.input_type == required_input_type, 'Not the good input type'
        self.input = self.load_input()


    def get_input_type(self, input_):
        if input_ == 'webcam':
            return Action.INPUT_TYPE_WEBCAM
        return  Action.INPUT_TYPE_FILE if isfile(input_) else Action.INPUT_TYPE_FOLDER


    def perform(self):
        raise NotImplementedError


    def show_result(self, result, title=None):
        if isinstance(result, list):
            titles = title if title is not None else [str(i) for i in range(len(result))]
            show_images(result, titles)
            return
        title = title if title is not None else 'Result'
        show_image(result, title)


    def load_input(self):
        if self.input_type == Action.INPUT_TYPE_WEBCAM:
            return []

        if self.input_type ==  Action.INPUT_TYPE_FILE:
            return load_image(self.args['input'])

        elif self.input_type ==  Action.INPUT_TYPE_FOLDER:
            return load_images(self.args['input'])

