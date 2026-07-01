import numpy as np

DTYPE = np.float32


class Tensor:
    """
    A minimal automatic differentiation engine built on NumPy.

    Every Tensor tracks two things that make backprop possible:
      - gradient_fn : a closure that knows how to push gradients
                      back to this tensor's parent tensors.
      - parents     : the set of Tensors that were inputs to the
                      operation that produced this tensor.

    Together they form a directed acyclic graph (DAG) called the
    computation graph.  Forward pass builds the graph; backward()
    traverses it in reverse to accumulate gradients.

    KEY CONCEPT — leaf vs. intermediate tensors:
      Leaf tensors (model parameters, inputs) are created directly
      via Tensor(data).  Their gradient_fn is a no-op and parents
      is empty.  Intermediate tensors are produced by operations
      (+, @, etc.) and have gradient_fn / parents set by
      attach_grad_fn().
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = np.zeros_like(self.data)
        # Default gradient_fn does nothing — overwritten for intermediate tensors.
        self.gradient_fn = lambda: None
        self.parents = set()

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """
        Run backpropagation from this tensor (the scalar loss).

        Algorithm: Kahn-style topological sort of the computation graph,
        then iterate in reverse order (output → input), calling each
        node's gradient_fn to propagate dL/d(node) to its parents.

        WHY topological sort?
          A node's gradient must be fully accumulated before we
          propagate it further.  Topo order guarantees every node is
          processed only after all its downstream nodes.

        WHY clear gradient_fn and parents after use?
          The closures capture references to intermediate Tensors,
          keeping the entire graph alive in memory.  Without clearing,
          each training step leaks one full graph and the process is
          eventually killed by the OS (OOM).  Clearing breaks the
          reference cycles so Python's garbage collector can reclaim
          the memory immediately.
        """
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t.parents:
                    build_topo(p)
                topo.append(t)

        build_topo(self)

        # Seed gradient: dL/dL = 1
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t.gradient_fn()
            # Release graph references immediately after use.
            t.gradient_fn = lambda: None
            t.parents = set()

    # ------------------------------------------------------------------
    # Primitive operations — each defines its own backward rule.
    # Pattern: compute forward result p, define gradient_fn that
    # adds dL/d(self) and dL/d(other) via the chain rule, then
    # attach it to p.
    # ------------------------------------------------------------------

    def __add__(self, other):
        p = Tensor(self.data + other.data)

        def gradient_fn():
            # d(self + other)/d(self) = 1, so dL/d(self) += dL/dp * 1
            # _unbroadcast handles the case where shapes were broadcast
            # during the forward pass — we must sum the gradient back
            # over the broadcast dimensions.
            self.grad += self._unbroadcast(p.grad, self.data.shape)
            other.grad += self._unbroadcast(p.grad, other.data.shape)

        return p.attach_grad_fn(gradient_fn, {self, other})

    def __sub__(self, other):
        p = Tensor(self.data - other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.data.shape)
            other.grad += self._unbroadcast(-p.grad, other.data.shape)

        return p.attach_grad_fn(gradient_fn, {self, other})

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            # Product rule: d(a*b)/da = b, d(a*b)/db = a
            self.grad += self._unbroadcast(p.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(p.grad * self.data, other.data.shape)

        return p.attach_grad_fn(gradient_fn, {self, other})

    def __truediv__(self, other):
        p = Tensor(self.data / other.data)

        def gradient_fn():
            # Quotient rule: d(a/b)/da = 1/b, d(a/b)/db = -a/b²
            self.grad += self._unbroadcast(p.grad / other.data, self.data.shape)
            other.grad += self._unbroadcast(-p.grad * self.data / (other.data ** 2), other.data.shape)

        return p.attach_grad_fn(gradient_fn, {self, other})

    def __matmul__(self, other):
        # self: (..., M, K)   other: (..., K, N)   p: (..., M, N)
        p = Tensor(np.matmul(self.data, other.data))

        def gradient_fn():
            # Matrix calculus:
            #   dL/d(self)  = dL/dp @ other^T    shape (..., M, K)
            #   dL/d(other) = self^T @ dL/dp      shape (..., K, N)
            # swapaxes(-1, -2) is a batched transpose that works for
            # 2-D and 3-D tensors (used in multi-head attention).
            self.grad += np.matmul(p.grad, other.data.swapaxes(-1, -2))
            other.grad += np.matmul(self.data.swapaxes(-1, -2), p.grad)

        return p.attach_grad_fn(gradient_fn, {self, other})

    def transpose(self, axes=None):
        p = Tensor(np.transpose(self.data, axes))

        def gradient_fn():
            if axes is None:
                self.grad += np.transpose(p.grad)
            else:
                # Inverse permutation: if forward applied axes=(1,0,2),
                # backward applies argsort([1,0,2]) = (1,0,2) again here,
                # but in general argsort gives the true inverse.
                idx = np.argsort(axes)
                self.grad += np.transpose(p.grad, idx)

        return p.attach_grad_fn(gradient_fn, {self})

    @property
    def T(self):
        return self.transpose()

    def concat(self, other, axis):
        p = Tensor(np.concatenate([self.data, other.data], axis=axis))

        def gradient_fn():
            # Split the gradient at the boundary where the two tensors
            # were joined in the forward pass.
            grad = np.split(p.grad, [self.data.shape[axis]], axis=axis)
            self.grad += grad[0]
            other.grad += grad[1]

        return p.attach_grad_fn(gradient_fn, {self, other})

    def reshape(self, shape):
        p = Tensor(np.reshape(self.data, shape))

        def gradient_fn():
            # Reshape is a view operation — gradient flows back with
            # the original shape restored.
            self.grad += np.reshape(p.grad, self.data.shape)

        return p.attach_grad_fn(gradient_fn, {self})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def shape(self):
        return self.data.shape

    def size(self):
        return np.prod(self.data.shape[1:])

    def attach_grad_fn(self, gradient_fn, parents):
        """
        Attach a backward function and the set of parent tensors to
        this tensor, turning it from a leaf into an intermediate node.
        Called at the end of every operation's forward pass.
        """
        self.gradient_fn = gradient_fn
        self.parents = parents
        return self

    @staticmethod
    def _unbroadcast(grad, shape):
        """
        Reverse NumPy broadcasting by summing gradient over the axes
        that were implicitly expanded during the forward pass.

        NumPy broadcast rules add dimensions on the left and repeat
        along any axis where the original size was 1.  To undo this:
          1. Sum over any extra leading dimensions.
          2. Sum (keeping dims) over axes where the original size was 1.
        """
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad.reshape(shape)

    def __str__(self):
        return str(self.data)