"""Datasets: synthetic blobs/moons + sklearn built-ins, all as (X, y) numpy."""

from __future__ import annotations

import numpy as np


def blobs(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = [(-2, -2), (2, 2), (-2, 2)]
    X = np.vstack([rng.normal(c, 0.9, (n // 3, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n // 3)
    return X, y


def moons(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    outer = np.stack([np.cos(t), np.sin(t)], 1)
    inner = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.vstack([outer, inner]) + rng.normal(0, 0.12, (n, 2))
    y = np.repeat([0, 1], n // 2)
    return X, y


def sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris()
    elif name == "breast-cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
    else:
        raise ValueError(f"unknown dataset: {name}")
    return d.data, d.target, list(d.target_names)


def load_csv(path: str, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col]).select_dtypes("number").to_numpy()
    return X, y
