"""Linear softmax classifier with hand-derived SGD.

Pure-Python lists, small dimensions, exact arithmetic order documented in the
code. Gradients: for logits z = W x + b and p = softmax(z), the cross-entropy
gradient is dL/dz = p - onehot(y), dL/dW = (p - onehot(y)) x^T, dL/db = p - onehot(y).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


class LinearClassifier:
    def __init__(self, n_classes: int, dim: int, rng: random.Random, scale: float = 0.01):
        self.n_classes = n_classes
        self.dim = dim
        self.W = [[rng.gauss(0, scale) for _ in range(dim)] for _ in range(n_classes)]
        self.b = [0.0] * n_classes

    def logits(self, x: Sequence[float]) -> List[float]:
        return [sum(wj * xj for wj, xj in zip(row, x)) + bj
                for row, bj in zip(self.W, self.b)]

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
            p = softmax(self.logits(x))
            total += -math.log(p[y] + 1e-12)
            dz = p[:]
            dz[y] -= 1.0
            for j in range(self.n_classes):
                for k in range(self.dim):
                    self.W[j][k] -= lr * dz[j] * x[k]
                self.b[j] -= lr * dz[j]
        return total / max(len(data), 1)
