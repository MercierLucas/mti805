import os
import cv2
import string
import numpy as np

from .classification_dataset import ClassifDataset


class CustomDataset(ClassifDataset):
    def __init__(self, data, train_split=1, limit=None):
        self.test_x = []
        self.labels = list(string.ascii_uppercase)
        emnist_classes = list(range(10)) + list (string.ascii_letters)
        labels = []
        for filename in os.listdir(data):
            img = cv2.imread(os.path.join(data, filename))
            if img is not None:
                img = self.preprocess(img)
                self.test_x.append(img)
                filename = filename.split('.')[0]
                labels.append(filename)

        self.test_x = np.array(self.test_x)
        self.test_y = np.array([self.labels.index(i)+1 for i in labels])
        self.test_y_emnist = np.array([emnist_classes.index(i) for i in labels])


    def emnist_class_to_label(self, idx):
        emnist = list(range(10)) + list (string.ascii_letters)
        return emnist[idx]


    def get_label(self, idx):
        try:
            return self.labels[idx-1]
        except:
            return self.emnist_class_to_label(idx)


    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        gray = cv2.resize(gray, (28, 28))
        gray = np.pad(gray, 2, mode='constant')
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
        gray = np.expand_dims(gray, axis=-1)
        return gray

        
