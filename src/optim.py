import math
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def reset(self):
        """Zero all parameter gradients before each forward-backward step."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def step(self):
        pass

    def clip_grad_norm(self, max_norm=1.0):
        """
        Global gradient norm clipping.

        Computes the L2 norm over ALL parameter gradients combined,
        then scales every gradient down by the same factor if the
        total norm exceeds max_norm.

        WHY global (not per-parameter)?
          Scaling all gradients by the same factor preserves the
          direction of the update — only the magnitude is clipped.
          Per-parameter clipping distorts the direction.

        WHY is this important for transformers?
          Attention scores can produce very large gradients when
          the model is confident but wrong.  Clipping prevents a
          single bad step from destabilising training.
        """
        total_sq = 0.0
        for p in self.parameters:
            if p is not None and p.grad is not None:
                total_sq += float(np.sum(p.grad.astype(np.float64) ** 2))

        total_norm = np.sqrt(total_sq)

        if total_norm > max_norm > 0:
            scale = max_norm / (total_norm + 1e-6)
            for p in self.parameters:
                if p is not None and p.grad is not None:
                    p.grad *= scale

        return total_norm


class AdamOptimizer(Optimizer):
    """
    Adam optimiser (Kingma & Ba, 2014).

    Maintains exponential moving averages of the gradient (m, first
    moment) and squared gradient (v, second moment) for each parameter.

    Update rule:
      m  ← β1*m + (1-β1)*g               # momentum
      v  ← β2*v + (1-β2)*g²              # adaptive scaling
      m̂  = m / (1-β1^t)                  # bias-corrected
      v̂  = v / (1-β2^t)                  # bias-corrected
      θ  ← θ - lr * m̂ / (√v̂ + ε)

    WHY bias correction?
      m and v are initialised to 0.  Early in training they are
      biased towards 0 because the sums haven't had time to warm up.
      Dividing by (1-β^t) corrects for this cold-start effect.

    WHY second moment (v)?
      It acts as a per-parameter learning rate: parameters with
      consistently large gradients get a smaller effective lr,
      while rarely updated parameters get a larger one.
    """

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [None for _ in range(len(parameters))]
        self.v = [None for _ in range(len(parameters))]
        self.t = 0      # global step counter for bias correction

    def step(self):
        self.t += 1

        for idx, p in enumerate(self.parameters):
            if p is not None and p.grad is not None:
                grad = p.grad.reshape(p.data.shape)

                # Lazy initialisation — allocate moment buffers on first step
                if self.m[idx] is None:
                    self.m[idx] = np.zeros_like(p.data)
                    self.v[idx] = np.zeros_like(p.data)

                # Update biased moment estimates
                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

                # Hook for AdamW (no-op in plain Adam)
                self._apply_weight_decay(p)

                # Bias-corrected estimates
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _apply_weight_decay(self, p):
        pass   # overridden in AdamW


class AdamWOptimizer(AdamOptimizer):
    """
    AdamW (Loshchilov & Hutter, 2019) — Adam with decoupled weight decay.

    WHY decouple weight decay from the gradient update?
      In plain Adam, adding L2 regularisation (λ||θ||²) to the loss
      means the weight decay term gets divided by √v̂, making the
      effective decay rate non-uniform across parameters.  AdamW
      applies weight decay directly to the parameters, independent
      of the adaptive scaling, which is the mathematically correct
      form of L2 regularisation under an adaptive optimiser.

    Implementation:
      θ ← θ * (1 - lr * λ)     # applied BEFORE the Adam step
      θ ← θ - lr * m̂ / (√v̂+ε) # Adam step as usual

    Weight decay should only be applied to 2-D weight matrices,
    NOT to biases or LayerNorm parameters (scale/shift).  Callers
    should exclude those from the parameter list passed to this class,
    or split into decay / no-decay groups.
    """

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps, weight_decay)

    def _apply_weight_decay(self, p):
        # Only shrink weight matrices (2-D), not biases or LayerNorm params.
        if self.weight_decay != 0.0 and p.data.ndim >= 2:
            p.data -= p.data * self.weight_decay * self.lr


class WarmupCosineScheduler:
    """
    Learning rate schedule used by GPT-3: linear warmup then cosine decay.

    Phase 1 — Warmup (steps 0 … warmup_steps):
      lr increases linearly from 0 to max_lr.
      WHY warmup?  At initialisation, gradients are noisy.  A large lr
      at step 0 can push parameters into bad regions that take many
      steps to recover from.  Starting small and ramping up lets the
      model find a reasonable direction first.

    Phase 2 — Cosine decay (warmup_steps … total_steps):
      lr follows a half-cosine from max_lr down to min_lr.
      WHY cosine?  It decays slowly at first (when the model is still
      making large progress) and flattens near min_lr towards the end
      (fine-tuning the solution).  Empirically better than linear decay
      for language models.
    """

    def __init__(self, max_lr, total_steps, warmup_steps, min_lr=0.0):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(min(warmup_steps, self.total_steps), 0)

    def lr_at(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / max(decay_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine