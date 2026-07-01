from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor, DTYPE


class Layer(ABC):

    def __init__(self):
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abstractmethod
    def forward(self, *args):
        pass

    @staticmethod
    def parameters():
        return []

    def __str__(self):
        return ''


class Sequential(Layer):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def __str__(self):
        return '\n'.join(str(l) for l in self.layers if str(l))


class Linear(Layer):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = Tensor(np.random.randn(out_size, in_size).astype(DTYPE) * np.sqrt(2 / in_size))
        self.bias = Tensor(np.zeros(out_size, dtype=DTYPE))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad += p.grad.T @ x.data
            self.bias.grad += np.sum(p.grad, axis=0)
            x.grad += p.grad @ self.weight.data

        return p.attach_grad_fn(gradient_fn, {self.weight, self.bias, x})

    def parameters(self):
        return [self.weight, self.bias]

    def __str__(self):
        return f'weight: {self.weight}\nbias: {self.bias}'


class Embedding(Layer):

    def __init__(self, vocabulary_size, embedding_size, axis=None):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.axis = axis

        self.weight = Tensor(
            np.random.randn(vocabulary_size, embedding_size).astype(DTYPE) * np.sqrt(2 / vocabulary_size)
        )

    def forward(self, x: Tensor):
        token_ids = x.data.astype(np.int64)
        weights = self.weight.data[token_ids]      
        p = Tensor(np.sum(weights, axis=self.axis) if self.axis is not None else weights)

        def gradient_fn():
            grad = p.grad
            if self.axis is not None:
                grad = np.expand_dims(grad, axis=self.axis)
                grad = np.broadcast_to(grad, weights.shape)

            np.add.at(self.weight.grad, token_ids, grad)

        return p.attach_grad_fn(gradient_fn, {self.weight})

    def parameters(self):
        return [self.weight]


class Dropout(Layer):

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x: Tensor):
        if not self.training or self.prob == 0:
            return x

        keep_prob = 1.0 - self.prob
        mask = (np.random.rand(*x.data.shape) < keep_prob).astype(DTYPE) / keep_prob
        p = Tensor(x.data * mask)

        def gradient_fn():
            x.grad += p.grad * mask

        return p.attach_grad_fn(gradient_fn, {x})


class Normalize(Layer):

    def __init__(self, size, eps=0.00001):
        super().__init__()
        self.size = size
        self.eps = eps

        self.scale = Tensor(np.ones(self.size, dtype=DTYPE))
        self.shift = Tensor(np.zeros(self.size, dtype=DTYPE))

    def forward(self, x: Tensor):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        norm = (x.data - mean) / np.sqrt(var + self.eps)
        p = Tensor(self.scale.data * norm + self.shift.data)

        def gradient_fn():
            axis = tuple(range(p.grad.ndim - 1))
            self.scale.grad += np.sum(p.grad * norm, axis=axis)
            self.shift.grad += np.sum(p.grad, axis=axis)

            grad = p.grad * self.scale.data
            grad_mean = np.mean(grad, axis=-1, keepdims=True)
            norm_mean = np.mean(grad * norm, axis=-1, keepdims=True)
            x.grad += (grad - grad_mean - norm * norm_mean) / np.sqrt(var + self.eps)

        return p.attach_grad_fn(gradient_fn, {self.scale, self.shift, x})

    def parameters(self):
        return [self.scale, self.shift]
