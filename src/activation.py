import numpy as np

from src.layer import Layer
from src.tensor import Tensor, DTYPE


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad += (p.data > 0) * p.grad

        return p.attach_grad_fn(gradient_fn, {x})


class GeLU(Layer):

    def forward(self, x: Tensor):
        k = np.sqrt(2.0 / np.pi)
        inner = k * (x.data + 0.044715 * x.data ** 3)
        tanh_val = np.tanh(inner)
        cdf = 0.5 * (1.0 + tanh_val)
        p = Tensor(x.data * cdf)

        def gradient_fn():
            sech2 = 1.0 - tanh_val ** 2
            dcdf = 0.5 * sech2 * k * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            x.grad += p.grad * (cdf + x.data * dcdf)

        return p.attach_grad_fn(gradient_fn, {x})


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

        return p.attach_grad_fn(gradient_fn, {x})


class Triu(Layer):

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones((x.data.shape[-2], x.data.shape[-1]), dtype=DTYPE))
        p = Tensor(np.where(keep == 1, x.data, np.array(self.value, dtype=DTYPE)))

        def gradient_fn():
            x.grad += p.grad * keep

        return p.attach_grad_fn(gradient_fn, {x})
