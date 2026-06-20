import math
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def reset(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def step(self):
        pass

    def clip_grad_norm(self, max_norm=1.0):
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

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [None for _ in range(len(parameters))]
        self.v = [None for _ in range(len(parameters))]
        self.t = 0

    def step(self):
        self.t += 1

        for idx, p in enumerate(self.parameters):
            if p is not None and p.grad is not None:
                grad = p.grad.reshape(p.data.shape)

                if self.m[idx] is None:
                    self.m[idx] = np.zeros_like(p.data)
                    self.v[idx] = np.zeros_like(p.data)

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                if self.weight_decay != 0.0:
                    p.data -= self.lr * self.weight_decay * p.data

                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class WarmupCosineScheduler:

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
