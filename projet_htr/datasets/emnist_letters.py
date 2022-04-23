import string
from .classification_dataset import ClassifDataset
from extra_keras_datasets import emnist


class EMNISTLettersDataset(ClassifDataset):
    def __init__(self, train_split=.7, limit=None):
        super().__init__(emnist.load_data(type='letters'), train_split, limit=limit)
        self.name = 'EMNIST_letters'
        self.labels = list (string.ascii_lowercase)


    def get_label(self, idx):
        return self.labels[idx]


    def get_idx_by_char(self, char):
        return self.labels.index(char)


    def get_char_sample(self, char, know_idx=None):
        idx = self.get_idx_by_char(char)
        if know_idx:
            x = self.train_x[know_idx]
            return x

        for i, (x, y) in enumerate(zip(self.train_x, self.train_y)):
            if y == idx:
                return x

        
