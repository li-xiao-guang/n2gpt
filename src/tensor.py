import numpy as np

DTYPE = np.float32


class Tensor:

    def __init__(self, data):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = np.zeros_like(self.data)
        self.gradient_fn = lambda: None
        self.parents = set()

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t.parents:
                    build_topo(p)
                topo.append(t)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t.gradient_fn()

    def shape(self):
        return self.data.shape

    def size(self):
        return np.prod(self.data.shape[1:])

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        p = Tensor(self.data + other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.data.shape)
            other.grad += self._unbroadcast(p.grad, other.data.shape)

        return p.update(gradient_fn, {self, other})

    def __sub__(self, other):
        p = Tensor(self.data - other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.data.shape)
            other.grad += self._unbroadcast(-p.grad, other.data.shape)

        return p.update(gradient_fn, {self, other})

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(p.grad * self.data, other.data.shape)

        return p.update(gradient_fn, {self, other})

    def __truediv__(self, other):
        p = Tensor(self.data / other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad / other.data, self.data.shape)
            other.grad += self._unbroadcast(-p.grad * self.data / (other.data ** 2), other.data.shape)

        return p.update(gradient_fn, {self, other})

    def __matmul__(self, other):
        p = Tensor(np.matmul(self.data, other.data))

        def gradient_fn():
            self.grad += np.matmul(p.grad, other.data.swapaxes(-1, -2))
            other.grad += np.matmul(self.data.swapaxes(-1, -2), p.grad)

        return p.update(gradient_fn, {self, other})

    def transpose(self, axes=None):
        p = Tensor(np.transpose(self.data, axes))

        def gradient_fn():
            if axes is None:
                self.grad += np.transpose(p.grad)
            else:
                idx = np.argsort(axes)
                self.grad += np.transpose(p.grad, idx)

        return p.update(gradient_fn, {self})

    @property
    def T(self):
        return self.transpose()

    def concat(self, other, axis):
        p = Tensor(np.concatenate([self.data, other.data], axis=axis))

        def gradient_fn():
            grad = np.split(p.grad, [self.data.shape[axis]], axis=axis)
            self.grad += grad[0]
            other.grad += grad[1]

        return p.update(gradient_fn, {self, other})

    def reshape(self, shape):
        p = Tensor(np.reshape(self.data, shape))

        def gradient_fn():
            self.grad += np.reshape(p.grad, self.data.shape)

        return p.update(gradient_fn, {self})

    def update(self, gradient_fn, parents):
        self.gradient_fn = gradient_fn
        self.parents = parents
        return self

    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad.reshape(shape)
