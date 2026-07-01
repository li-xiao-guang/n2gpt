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
    """
    Cross-entropy loss with a fused softmax step.

    WHY fuse softmax + CE instead of applying Softmax as a separate
    layer?
      The naive gradient of log(softmax(x)) involves subtracting a
      large positive from another large positive, which loses precision.
      When fused, the gradient simplifies to (softmax - y_onehot) / N,
      which is numerically clean and requires no log-of-softmax
      computation at all.

    Forward:
      softmax_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))   [stable]
      L = -Σ y_i * log(softmax_i) / N                         [CE]

    Backward (combined gradient w.r.t. raw logits):
      dL/dx_i = (softmax_i - y_i) / N

    y is expected to be a one-hot matrix of shape (T, vocab_size).
    N = T (number of token positions) — we average the loss over positions.
    """

    def loss(self, p: Tensor, y: Tensor):
        # Numerically stable softmax
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        log = np.log(np.clip(softmax, 1e-10, 1))   # clip avoids log(0)
        ce = Tensor(0 - np.sum(y.data * log) / len(y.data))

        def gradient_fn():
            # Combined softmax + CE gradient: (softmax - y) / N
            # This is the result of applying the chain rule through
            # both the log and the softmax in one step.
            p.grad += ce.grad * (softmax - y.data) / len(y.data)

        return ce.attach_grad_fn(gradient_fn, {p})


class BCELoss(Loss):
    """
    Binary cross-entropy loss for two-class (0/1) targets.

    L = -mean( y*log(p) + (1-y)*log(1-p) )

    The clipping prevents log(0) and the division by (p*(1-p)) in the
    gradient from blowing up.
    """

    def loss(self, p: Tensor, y: Tensor):
        clipped = np.clip(p.data, 1e-7, 1 - 1e-7)
        bce = Tensor(-np.mean(y.data * np.log(clipped) + (1 - y.data) * np.log(1 - clipped)))

        def gradient_fn():
            # dL/dp = (p - y) / (p * (1-p)) / N
            p.grad += (clipped - y.data) / (clipped * (1 - clipped)) / len(y.data)

        return bce.attach_grad_fn(gradient_fn, {p})