"""Pipeline: split, scale, train, cross-validate, evaluate, compare vs sklearn."""

from __future__ import annotations

import numpy as np

from classifiers import GaussianNB, KNN
from logreg import LogisticRegression
from metrics import report


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


class StandardScaler:
    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OneVsRest:
    """Wrap a binary classifier for multiclass."""

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs
        self.models_: list = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRest":
        self.classes_ = np.unique(y)
        self.models_ = []
        for c in self.classes_:
            m = self.cls(**self.kwargs).fit(X, (y == c).astype(int))
            self.models_.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.models_) == 2 and hasattr(self.models_[0], "predict_proba"):
            return (self.models_[1].predict_proba(X) >= 0.5).astype(int)
        scores = np.stack([m.predict_proba(X) if hasattr(m, "predict_proba")
                           else m.predict(X).astype(float)
                           for m in self.models_], axis=1)
        return self.classes_[scores.argmax(axis=1)]


def cross_validate(make_model, X: np.ndarray, y: np.ndarray, k: int = 5,
                   seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    accs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        model = make_model().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(float(np.mean(pred == y[te])))
    return {"cv_accuracy_mean": round(float(np.mean(accs)), 4),
            "cv_accuracy_std": round(float(np.std(accs)), 4)}


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    return report(y_test, model.predict(X_test))


def compare_models(X_train, y_train, X_test, y_test) -> list[dict]:
    """From-scratch models vs their sklearn counterparts."""
    from sklearn.linear_model import LogisticRegression as SkLogReg
    from sklearn.naive_bayes import GaussianNB as SkGNB
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    n_classes = len(np.unique(y_train))
    results = []
    pairs = [
        ("logreg (scratch)", lambda: OneVsRest(LogisticRegression, lr=0.5, epochs=800)
         if n_classes > 2 else LogisticRegression(lr=0.5, epochs=800),
         lambda: SkLogReg(max_iter=1000)),
        ("knn k=5 (scratch)", lambda: KNN(k=5), lambda: SkKNN(n_neighbors=5)),
        ("gaussian-nb (scratch)", lambda: GaussianNB(), lambda: SkGNB()),
    ]
    for name, make_mine, make_sk in pairs:
        mine = make_mine().fit(X_train, y_train)
        sk = make_sk().fit(X_train, y_train)
        r_mine = evaluate(mine, X_test, y_test)
        r_sk = evaluate(sk, X_test, y_test)
        results.append({"model": name, "accuracy": r_mine["accuracy"],
                        "f1": r_mine["f1"]})
        results.append({"model": name.replace("(scratch)", "(sklearn)"),
                        "accuracy": r_sk["accuracy"], "f1": r_sk["f1"]})
    return results
