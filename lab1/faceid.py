from face_embedder import FaceEmbedder
from face_encoder import FaceEncoder

class FaceID:

    def __init__(self, weights_root_path, pickle_path=None) -> None:
        self.embedder = FaceEmbedder(weights_root_path, pickle_path)
        self.encoder = FaceEncoder(pickle_path)
        

    def train(self, dataset):
        self.embedder.extract_from_dataset(dataset)
        self.encoder.train(self.embedder.embeddings, self.embedder.labels)

    
    def save(self, pickle_path):
        self.embedder.save(pickle_path)
        print('Embeddings saved')
        self.encoder.save(pickle_path)
        print('Face encoder saved')


    def recognize(self, image):
        embeddings = self.embedder.forward(image)
        prob, label_pred = self.encoder.identify(embeddings)
        return prob, label_pred
