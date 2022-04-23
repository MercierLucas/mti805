import pickle
from datasets import ClassifDataset

class Classifier:
    model = None
    name = ''
    labels = []

    def __init__(self, name, model=None) -> None:
        self.name = name
        self.accuracy = None

        if model:
            self.model = self.load(model)


    def load(self, model):
        with open(model, 'rb') as f:
            return pickle.load(f)


    def save(self, model):
        with open(model, 'wb') as f:
            pickle.dump(self.model, f)


    def train(self, data:ClassifDataset, save=False):
        pass


    def predict(self, image, return_class=False):
        pass


    def evaluate(self, data:ClassifDataset):
        pass