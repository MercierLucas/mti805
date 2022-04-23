from .classification_dataset import ClassifDataset
from tensorflow.keras.datasets import mnist


class MNISTDataset(ClassifDataset):
    def __init__(self, train_split=.7, limit=None):
        super().__init__(mnist.load_data(), train_split, limit=limit)
        self.labels = list(range(10))
        self.name = 'MNIST'


    def get_label(self, idx):
        return self.labels[idx]

        
