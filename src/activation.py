import numpy as np

from src.layer import Layer
from src.tensor import Tensor, DTYPE


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad += (p.data > 0) * p.grad

        return p.update(gradient_fn, {x})


class Softmax(Layer):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            grad = np.sum(p.data * p.grad, axis=self.axis, keepdims=True)
            x.grad += p.data * (p.grad - grad)

        return p.update(gradient_fn, {x})


class Triu(Layer):

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones((x.data.shape[-2], x.data.shape[-1]), dtype=DTYPE))

        p = Tensor(x.data)
        p.data = np.where(keep == 1, p.data, np.array(self.value, dtype=DTYPE))

        def gradient_fn():
            x.grad += p.grad * keep

        return p.update(gradient_fn, {x})
