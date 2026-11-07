"""Low-rank adaptation (LoRA-style) for a frozen LinearClassifier.

The adapter keeps the base weights W frozen and learns a delta W_delta = B @ A
with A (rank x dim) and B (n_classes x rank). B is zero-initialized so the
adapter starts as the identity adaptation. Gradients through the factorization,
with h = A x and z = W x + B h + b (b frozen):

  dL/dB = dz h^T
  dL/dA = (B^T dz) x^T

merge() returns W + B @ A. Merged logits reproduce adapter logits to within
floating-point reassociation error (max abs diff ~1e-14 on the checked-in
fixture, pinned in tests): adapter computes W x + B (A x), merged computes
(W + B @ A) x, and the reassociation is the only difference.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from .model import LinearClassifier, softmax


class LoRAClassifier:
    def __init__(self, base: LinearClassifier, rank: int, rng: random.Random,
                 scale: float = 0.01):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.rank = rank
        self.A = [[rng.gauss(0, scale) for _ in range(base.dim)] for _ in range(rank)]
        self.B = [[0.0] * rank for _ in range(base.n_classes)]

    def trainable_params(self) -> int:
        return self.rank * self.base.dim + self.base.n_classes * self.rank

    def logits(self, x: Sequence[float]) -> List[float]:
        h = [sum(ak * xk for ak, xk in zip(row, x)) for row in self.A]
        return [sum(wj * xj for wj, xj in zip(row, x)) + bj
                + sum(bjr * hr for bjr, hr in zip(brow, h))
                for row, bj, brow in zip(self.base.W, self.base.b, self.B)]

    def predict(self, x: Sequence[float]) -> int:
        ls = self.logits(x)
        return max(range(len(ls)), key=lambda i: (ls[i], -i))

    def accuracy(self, data: Sequence[Tuple[List[float], int]]) -> float:
        if not data:
            return 0.0
        return sum(1 for x, y in data if self.predict(x) == y) / len(data)

    def train_epoch(self, data: Sequence[Tuple[List[float], int]], lr: float) -> float:
        total = 0.0
        for x, y in data:
            h = [sum(ak * xk for ak, xk in zip(row, x)) for row in self.A]
            p = softmax(self.logits(x))
            total += -math.log(p[y] + 1e-12)
            dz = p[:]
            dz[y] -= 1.0
            # dL/dB = dz h^T ; dL/dA = (B^T dz) x^T (gradients at current weights)
            dA = [[sum(self.B[j][r] * dz[j] for j in range(self.base.n_classes)) * xk
                   for k, xk in enumerate(x)] for r in range(self.rank)]
            for j in range(self.base.n_classes):
                for r in range(self.rank):
                    self.B[j][r] -= lr * dz[j] * h[r]
            for r in range(self.rank):
                for k in range(self.base.dim):
                    self.A[r][k] -= lr * dA[r][k]
        return total / max(len(data), 1)

    def merged_weights(self) -> List[List[float]]:
        """W + B @ A, computed row by row."""
        return [[w + sum(self.B[j][r] * self.A[r][k] for r in range(self.rank))
                 for k, w in enumerate(row)]
                for j, row in enumerate(self.base.W)]

    def merged_logits(self, x: Sequence[float]) -> List[float]:
        Wm = self.merged_weights()
        return [sum(wj * xj for wj, xj in zip(row, x)) + bj
                for row, bj in zip(Wm, self.base.b)]
