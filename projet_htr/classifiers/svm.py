from copyreg import pickle


import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

from .classifier import Classifier
from datasets import ClassifDataset



class SVM(Classifier):
    def __init__(self, kernel, model=None) -> None:
        super().__init__(f'SVC{kernel}', model=model)
        if model is None:
            self.model = SVC(kernel=kernel)


    def train(self, data: ClassifDataset):
        x = data.train_flatten_images()
        self.model.fit(x, data.train_y)


    def predict(self, data):
        x = data.flatten()
        y_pred = self.model.predict([x])
        return y_pred[0]


    def evaluate(self, data: ClassifDataset):
        x = data.test_flatten_images()
        y_pred = self.model.predict(x)
        self.accuracy = metrics.accuracy_score(y_true=data.test_y, y_pred=y_pred)
        return self.accuracy