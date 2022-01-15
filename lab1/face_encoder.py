import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

ENCODER_FILE = 'face_encoder.pkl'

class FaceEncoder:

    def __init__(self, pickle_path=None) -> None:
        if pickle_path is not None:
            with open(os.path.join(pickle_path, ENCODER_FILE), 'rb') as f:
                encoder = pickle.load(f)
                self.label_encoder = encoder['label_encoder']
                self.face_classifier = encoder['face_classifier']


    def identify(self, embedding):
        preds = self.face_classifier.predict_proba(embedding)[0]
        j = np.argmax(preds)
        prob = preds[j]
        name = self.label_encoder.classes_[j]
        return prob, name


    def train(self, embeddings, labels):
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)
        self.face_classifier = SVC(C=1.0, kernel='linear', probability=True)
        self.face_classifier.fit(embeddings, labels)


    def save(self, pickle_path):
        with open(os.path.join(pickle_path, ENCODER_FILE), 'wb') as f:
            encoder = {'label_encoder': self.label_encoder, 'face_classifier': self.face_classifier}
            pickle.dump(encoder, f)