"""k-NN and Gaussian Naive Bayes from scratch, NumPy only."""

from __future__ import annotations

import numpy as np


class KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        # (m, n) squared-euclidean via broadcasting
        dists = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        nearest = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        votes = self.y[nearest]
        # majority vote per row
        return np.array([np.bincount(v.astype(int)).argmax() for v in votes])


class GaussianNB:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.theta_ = {}   # means per class
        self.sigma_ = {}   # variances per class
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(X)
            self.theta_[c] = Xc.mean(axis=0)
            self.sigma_[c] = Xc.var(axis=0) + 1e-9
        return self

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray,
                      var: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.stack([
            np.log(self.priors_[c]) + self._log_gaussian(X, self.theta_[c], self.sigma_[c])
            for c in self.classes_
        ], axis=1)
        return self.classes_[scores.argmax(axis=1)]
