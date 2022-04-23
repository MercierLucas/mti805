import cv2
import string
import numpy as np
import tensorflow as tf
from sklearn import metrics
from .classifier import Classifier
from datasets import ClassifDataset


class LeNet(Classifier):
    def __init__(self, model_path, name='lenet', classif_type='numbers') -> None:
        super().__init__(name, model=model_path)
        if classif_type == 'numbers':
            self.labels = list(range(10))
        elif classif_type == 'letters':
            self.labels = list (string.ascii_uppercase)
        elif classif_type == 'both':
            self.labels = list(range(10)) + list(string.ascii_letters)


    def train(self, data: ClassifDataset):
        pass


    def load(self, model):
        return tf.keras.models.load_model(model)


    def predict(self, image, return_class=False):
        if len(image.shape) != 4:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
        predicted = self.model.predict_classes(image)[0]
        if return_class:
            return self.labels[predicted]
        return predicted


    def evaluate(self, data: ClassifDataset):
        loss, acc = self.model.evaluate(data.test_x, data.test_y)
        self.accuracy = acc
        return acc
        #except:
        #preds = [self.predict(np.array([x])) for x in data.test_x]
        #self.accuracy = metrics.accuracy_score(y_true=data.test_y, y_pred=preds)
        #return self.accuracy
