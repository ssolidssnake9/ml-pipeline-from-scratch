"""Logistic regression from scratch: vectorized gradient descent, NumPy only."""

from __future__ import annotations

import numpy as np


class LogisticRegression:
    """Binary logistic regression. Multiclass via one-vs-rest in Pipeline."""

    def __init__(self, lr: float = 0.1, epochs: int = 1000,
                 l2: float = 0.0, seed: int = 0, verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0
        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if self.verbose and epoch % 200 == 0:
                loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
                self.loss_history.append(float(loss))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
