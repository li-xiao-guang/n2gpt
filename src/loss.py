from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor


class Loss(ABC):

    def __call__(self, p: Tensor, y: Tensor):
        return self.loss(p, y)

    @abstractmethod
    def loss(self, p: Tensor, y: Tensor):
        pass


class CELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        log = np.log(np.clip(softmax, 1e-10, 1))
        ce = Tensor(0 - np.sum(y.data * log) / len(y.data))

        def gradient_fn():
            p.grad += (softmax - y.data) / len(y.data)

        ce.gradient_fn = gradient_fn
        ce.parents = {p}
        return ce
