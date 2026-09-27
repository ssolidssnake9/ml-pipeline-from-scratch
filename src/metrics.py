"""Classification metrics from scratch (binary + multiclass, macro-averaged)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm, classes


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _prf_binary(y_true: np.ndarray, y_pred: np.ndarray, pos: int = 1):
    tp = int(np.sum((y_pred == pos) & (y_true == pos)))
    fp = int(np.sum((y_pred == pos) & (y_true != pos)))
    fn = int(np.sum((y_pred != pos) & (y_true == pos)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = "macro") -> dict[str, float]:
    """Macro (default) or micro averaged precision/recall/F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    if average == "micro":
        tp = fp = fn = 0
        for c in classes:
            yt = (y_true == c).astype(int)
            yp = (y_pred == c).astype(int)
            tp += int(np.sum((yp == 1) & (yt == 1)))
            fp += int(np.sum((yp == 1) & (yt == 0)))
            fn += int(np.sum((yp == 0) & (yt == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1}
    ps, rs, fs = [], [], []
    for c in classes:
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        p, r, f = _prf_binary(yt, yp)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return {"precision": float(np.mean(ps)), "recall": float(np.mean(rs)),
            "f1": float(np.mean(fs))}


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {"accuracy": round(accuracy(y_true, y_pred), 4)}
    out.update({k: round(v, 4) for k, v in
                precision_recall_f1(y_true, y_pred).items()})
    cm, classes = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()
    out["classes"] = classes.tolist()
    return out
