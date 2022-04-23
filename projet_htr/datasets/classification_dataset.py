import numpy as np
import tensorflow as tf


class ClassifDataset:
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    labels = []
    name = ''


    def __init__(self, data, train_split, limit=None):
      self.train_split = train_split
      self.curr_idx = 0

      train, test = data

      if limit:
        n_occurences = self.min_occuring_label(train[1]) if limit == 'smallest' else limit
        train = self.keep_n_occurences(train, n_occurences)

        n_occurences = self.min_occuring_label(test[1]) if limit == 'smallest' else limit
        test = self.keep_n_occurences(test, n_occurences)

      print(f'Train shape: {train[0].shape}')

      self.split_train_val(train[0], train[1], train_split)
      self.test_x = test[0]
      self.test_y = test[1]

      self.preprocess()


    def min_occuring_label(self, data):
      u, counts = np.unique(data, return_counts=True)
      stats = dict(zip(u, counts))
      return np.min(counts) #min(stats, key=stats.get)#min(stats.items(), key=lambda x: x[1]) 


    def keep_n_occurences(self, data, n_occurences):
      x, y = data
      x_kept, y_kept = [], []

      kept_occurences = {k: 0 for k in np.unique(y)}

      for idx, label in enumerate(y):
        if kept_occurences[label] > n_occurences:
          continue
        
        x_kept.append(x[idx])
        y_kept.append(y[idx])
        kept_occurences[label] += 1

      return np.array(x_kept), np.array(y_kept)


    def split_train_val(self, x, y, train_percentage):
      n_train = int(x.shape[0] * train_percentage)
      self.train_x = x[:n_train]
      self.train_y = y[:n_train]
      self.val_x = x[n_train:]
      self.val_y = y[n_train:]
      assert x.shape[0] == self.train_x.shape[0] + self.val_x.shape[0], 'Data missing'


    def _preprocess(self, serie):
      """Apply padding to convert 28x28 to 32x32 and add a dim (because of grayscale only 1channel)"""
      serie = tf.pad(serie, [[0, 0], [2, 2], [2, 2]]) / 255
      serie = tf.expand_dims(serie, axis=3)
      return serie


    def preprocess(self):
      self.train_x = np.array(self._preprocess(self.train_x))
      self.val_x = np.array(self._preprocess(self.val_x))
      self.test_x = np.array(self._preprocess(self.test_x))


    def train_flatten_images(self):
      x = np.array(self.train_x)
      return x.reshape([x.shape[0], -1])

    def test_flatten_images(self):
      x = np.array(self.test_x)
      return x.reshape([x.shape[0], -1])

    def get_as_np(self):
      return np.array(self.x)


    def get_label(self):
      return NotImplementedError


    def __getitem__(self, idx):
      return self.x[idx], self.y[idx]


    def __next__(self):
      self.curr_idx += 1

      if self.curr_idx == len(self.y):
          raise StopIteration
      return self[self.curr_idx]


    def __iter__(self):
      self.curr_idx = -1
      return self