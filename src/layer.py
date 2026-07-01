from abc import ABC, abstractmethod

import numpy as np

from src.tensor import Tensor, DTYPE


class Layer(ABC):
    """
    Abstract base class for all layers.

    A layer is callable (__call__ → forward) and knows its training
    mode (affects Dropout).  Subclasses expose their learnable
    parameters through parameters().
    """

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
    """
    Chains a list of layers into a single layer whose forward pass
    runs each sub-layer in order.

    IMPORTANT: Several GPT sub-modules (GPTAttention, GPTFeedForward)
    inherit Sequential but override forward().  In those cases the
    layers list is NOT used for the forward pass — it is only kept
    so that parameters() can recursively collect all learnable
    tensors without each subclass having to override parameters().
    """

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
    """
    Fully-connected (affine) layer:  p = x @ W^T + b

    Weight shape is (out, in) — transposed relative to the
    mathematical convention — so the forward pass is a simple
    row-vector times matrix:  (T, in) @ (in, out) = (T, out).

    Gradient derivation for W (no batch dimension, x is 2-D):
      Loss L is a scalar.
      p = x W^T  →  dL/dW = (dL/dp)^T @ x      shape (out, in)
      p = x W^T  →  dL/dx = (dL/dp) @ W         shape (T, in)
      dL/db = sum over the T dimension of dL/dp  shape (out,)
    """

    def __init__(self, in_size, out_size, std=0.02):
        super().__init__()
        # GPT-3 uses N(0, 0.02) for most weights.
        # Residual output projections are initialised with a smaller std
        # (0.02 / sqrt(2 * num_layers)) passed in via the std argument.
        self.weight = Tensor(np.random.randn(out_size, in_size).astype(DTYPE) * std)
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
    """
    Lookup table: maps integer token IDs to dense vectors.

    weight shape: (vocabulary_size, embedding_size)
    This matches Linear.weight convention, which is required for
    weight tying (GPT's output projection reuses this matrix).

    Forward:  simply index the weight matrix by token IDs.

    Backward: np.add.at performs a scatter-add — the gradient for
    each row is the sum of all upstream gradients that referenced
    that vocabulary index.  Normal indexing (weight.grad[ids] += g)
    does NOT accumulate correctly when the same id appears multiple
    times in a batch; np.add.at handles duplicates properly.
    """

    def __init__(self, vocabulary_size, embedding_size, axis=None):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.axis = axis

        # GPT-3 initialises embeddings with a smaller std (0.01) than
        # weight matrices (0.02) to keep early activations well-scaled.
        self.weight = Tensor(
            np.random.randn(vocabulary_size, embedding_size).astype(DTYPE) * 0.01
        )

    def forward(self, x: Tensor):
        token_ids = x.data.astype(np.int64)
        weights = self.weight.data[token_ids]       # (T, embedding_size)
        p = Tensor(np.sum(weights, axis=self.axis) if self.axis is not None else weights)

        def gradient_fn():
            grad = p.grad
            if self.axis is not None:
                grad = np.expand_dims(grad, axis=self.axis)
                grad = np.broadcast_to(grad, weights.shape)
            # scatter-add: accumulate gradient into the rows that were looked up
            np.add.at(self.weight.grad, token_ids, grad)

        return p.attach_grad_fn(gradient_fn, {self.weight})

    def parameters(self):
        return [self.weight]


class Dropout(Layer):
    """
    Inverted dropout: randomly zeroes activations during training and
    scales survivors up by 1 / keep_prob.

    WHY scale up (inverted)?
      At inference we use eval() which disables dropout.  Scaling
      during training means the expected value of each activation is
      the same at train and eval time — no correction needed at eval.
      The alternative (scale down at eval) would require touching the
      model at inference, which is less convenient.
    """

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x: Tensor):
        if not self.training or self.prob == 0:
            return x     # pass-through at eval time

        keep_prob = 1.0 - self.prob
        mask = (np.random.rand(*x.data.shape) < keep_prob).astype(DTYPE) / keep_prob
        p = Tensor(x.data * mask)

        def gradient_fn():
            # Same mask is applied to the gradient — zeroed neurons
            # contribute nothing, surviving neurons are scaled the same way.
            x.grad += p.grad * mask

        return p.attach_grad_fn(gradient_fn, {x})


class Normalize(Layer):
    """
    Layer Normalization — normalises each token's feature vector
    independently (across the embedding dimension, axis=-1).

    Unlike BatchNorm (normalises across samples in a batch),
    LayerNorm works on a single sequence and has no dependency on
    batch size or other tokens, which makes it well-suited for
    autoregressive language modelling.

    Forward:
      norm = (x - mean) / sqrt(var + eps)
      out  = scale * norm + shift         # per-feature affine rescale

    Backward (the full analytic gradient — not trivial to derive):
      Let N = size of the last dimension.
      dL/dx = (1/sqrt(var+eps)) * (dL/d_norm
                - mean(dL/d_norm)
                - norm * mean(dL/d_norm * norm))
      where dL/d_norm = dL/dp * scale.

    The three-term formula corrects for the mean and variance
    being functions of x themselves (implicit dependencies via the
    normalisation denominator).
    """

    def __init__(self, size, eps=0.00001):
        super().__init__()
        self.size = size
        self.eps = eps
        # scale (γ) and shift (β) initialised to 1 and 0 so that
        # LayerNorm starts as an identity transformation.
        self.scale = Tensor(np.ones(self.size, dtype=DTYPE))
        self.shift = Tensor(np.zeros(self.size, dtype=DTYPE))

    def forward(self, x: Tensor):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        norm = (x.data - mean) / np.sqrt(var + self.eps)
        p = Tensor(self.scale.data * norm + self.shift.data)

        def gradient_fn():
            # Gradient for affine parameters — sum over all positions
            # (all axes except the last feature axis).
            axis = tuple(range(p.grad.ndim - 1))
            self.scale.grad += np.sum(p.grad * norm, axis=axis)
            self.shift.grad += np.sum(p.grad, axis=axis)

            # Gradient for x — the three-term formula.
            grad = p.grad * self.scale.data
            grad_mean = np.mean(grad, axis=-1, keepdims=True)
            norm_mean = np.mean(grad * norm, axis=-1, keepdims=True)
            x.grad += (grad - grad_mean - norm * norm_mean) / np.sqrt(var + self.eps)

        return p.attach_grad_fn(gradient_fn, {self.scale, self.shift, x})

    def parameters(self):
        return [self.scale, self.shift]