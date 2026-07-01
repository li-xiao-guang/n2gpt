import numpy as np

from src.layer import Layer
from src.tensor import Tensor, DTYPE


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def gradient_fn():
            # d(tanh)/dx = 1 - tanh²(x) = 1 - p²
            # We reuse p.data (already computed) instead of calling tanh again.
            x.grad += p.grad * (1 - p.data ** 2)

        return p.attach_grad_fn(gradient_fn, {x})


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            # Gradient is 1 where the forward pass was positive, 0 elsewhere.
            # Using p.data > 0 (not x.data > 0) is equivalent but avoids
            # re-reading x after it may have been overwritten.
            x.grad += (p.data > 0) * p.grad

        return p.attach_grad_fn(gradient_fn, {x})


class GeLU(Layer):
    """
    Gaussian Error Linear Unit — the activation used in GPT-3.

    Exact definition:  GeLU(x) = x * Φ(x)
    where Φ is the standard normal CDF.

    WHY GeLU over ReLU?
      ReLU hard-zeros negative inputs, which can kill gradient flow
      (dead neurons).  GeLU smoothly weights inputs by their
      probability of being positive, giving non-zero gradient almost
      everywhere and leading to better training for transformers.

    Tanh approximation (used here, same as GPT-2/3):
      GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    KEY: tanh_val and cdf are computed once in the forward pass and
    captured by the closure.  The backward pass reuses them directly —
    do not recompute, as x.data may differ by then (next forward step).
    """

    def forward(self, x: Tensor):
        k = np.sqrt(2.0 / np.pi)
        inner = k * (x.data + 0.044715 * x.data ** 3)
        tanh_val = np.tanh(inner)
        cdf = 0.5 * (1.0 + tanh_val)
        p = Tensor(x.data * cdf)

        def gradient_fn():
            # d(GeLU)/dx = cdf(x) + x * d(cdf)/dx
            #
            # d(cdf)/dx = 0.5 * sech²(inner) * d(inner)/dx
            #           = 0.5 * (1 - tanh²) * k * (1 + 3 * 0.044715 * x²)
            sech2 = 1.0 - tanh_val ** 2
            dcdf = 0.5 * sech2 * k * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            x.grad += p.grad * (cdf + x.data * dcdf)

        return p.attach_grad_fn(gradient_fn, {x})


class Sigmoid(Layer):

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        # Clip before exp to prevent overflow for large negative inputs.
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def gradient_fn():
            # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
            x.grad += p.grad * p.data * (1 - p.data)

        return p.attach_grad_fn(gradient_fn, {x})


class Softmax(Layer):
    """
    Numerically stable softmax: subtract the row maximum before exp.

    Without this, exp(large_number) overflows to inf.  Subtracting
    the max does not change the output (it cancels in numerator and
    denominator) but keeps all exp() calls on non-positive inputs.
    """

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            # Full Jacobian of softmax is an NxN matrix, but we never
            # materialise it.  The vector-Jacobian product simplifies to:
            #   dL/dx_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
            # which is the dot product of the incoming gradient with p,
            # broadcast and subtracted — O(N) instead of O(N²).
            grad = np.sum(p.data * p.grad, axis=self.axis, keepdims=True)
            x.grad += p.data * (p.grad - grad)

        return p.attach_grad_fn(gradient_fn, {x})


class Triu(Layer):
    """
    Causal (autoregressive) mask applied to attention score matrices.

    Sets upper-triangle entries to -1e9 so that after softmax they
    become ~0, preventing position i from attending to future
    positions j > i.  This is what makes GPT "causal" — during
    training every token position can only see its own past.

    Shape note: x is (H, T, T) for multi-head attention.
    The mask is built as (T, T) and broadcast over the head dim.
    """

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones((x.data.shape[-2], x.data.shape[-1]), dtype=DTYPE))
        p = Tensor(np.where(keep == 1, x.data, np.array(self.value, dtype=DTYPE)))

        def gradient_fn():
            # Gradient flows only through the positions that were kept
            # (lower triangle).  Masked positions had a constant value
            # in the forward pass, so their gradient is 0.
            x.grad += p.grad * keep

        return p.attach_grad_fn(gradient_fn, {x})