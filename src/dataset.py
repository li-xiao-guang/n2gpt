from abc import ABC, abstractmethod

from src.tensor import Tensor


class Dataset(ABC):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

        self.test_labels = None
        self.test_features = None
        self.train_labels = None
        self.train_features = None
        self.labels = None
        self.features = None

        self.load()
        self.train()

    @abstractmethod
    def load(self):
        pass

    def train(self):
        self.features = self.train_features
        self.labels = self.train_labels

    def eval(self):
        self.features = self.test_features
        self.labels = self.test_labels

    def shape(self):
        return Tensor(self.features).size(), Tensor(self.labels).size()

    def items(self):
        return Tensor(self.features), Tensor(self.labels)

    def __len__(self):
        return len(self.features) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        return Tensor(self.features[start: end]), Tensor(self.labels[start: end])

    @abstractmethod
    def estimate(self, predictions):
        pass
