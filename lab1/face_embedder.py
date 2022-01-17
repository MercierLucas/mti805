import os
import cv2
import pickle


WEIGHT_FILE = 'openface.nn4.small2.v1.t7'
EMBEDDING_FILE = 'embedding.pkl'


class FaceEmbedder:

    def __init__(self, weights_root_path, pickle_path=None) -> None:
        self.embedder = cv2.dnn.readNetFromTorch(f'{weights_root_path}/{WEIGHT_FILE}')
        self.labels = []
        self.embeddings = []

        if pickle_path is not None:
            with open(os.path.join(pickle_path, EMBEDDING_FILE), 'rb') as f:
                known = pickle.load(f)
                self.embeddings = known['embeddings']
                self.labels = known['labels']


    def forward(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=False, crop=False)
        self.embedder.setInput(blob)
        embedding = self.embedder.forward()
        return embedding


    def extract_from_dataset(self, dataset):
        for image, label in dataset:
            self.extract_from_image(image, label)


    def extract_from_image(self, image, label=None):
        embedding = self.forward(image)
        if label is not None:
            self.embeddings.append(embedding.flatten())
            self.labels.append(label)


    def save(self, pickle_path):
        with open(os.path.join(pickle_path, EMBEDDING_FILE), 'wb') as f:
            known = {'embeddings': self.embeddings, 'labels': self.labels}
            pickle.dump(known, f)




    

    