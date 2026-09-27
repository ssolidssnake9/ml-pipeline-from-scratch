"""CLI: run a model on a dataset, or compare from-scratch vs sklearn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifiers import GaussianNB, KNN  # noqa: E402
from logreg import LogisticRegression  # noqa: E402
from datasets import blobs, load_csv, moons, sklearn_dataset  # noqa: E402
from metrics import report  # noqa: E402
from pipeline import (OneVsRest, StandardScaler, compare_models,  # noqa: E402
                      cross_validate, train_test_split)


def get_dataset(name: str, csv: str | None, target: str | None):
    if csv:
        return load_csv(csv, target)
    if name == "blobs":
        return blobs()
    if name == "moons":
        return moons()
    X, y, _ = sklearn_dataset(name)
    return X, y


def make_model(name: str, n_classes: int):
    if name == "logreg":
        base = lambda: LogisticRegression(lr=0.5, epochs=800)  # noqa: E731
        return OneVsRest(LogisticRegression, lr=0.5, epochs=800) if n_classes > 2 else base()
    if name == "knn":
        return KNN(k=5)
    if name == "nb":
        return GaussianNB()
    raise ValueError(f"unknown model: {name}")


def cmd_run(args):
    X, y = get_dataset(args.dataset, args.csv, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=args.seed)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = make_model(args.model, len(set(y_train.tolist())))
    model.fit(X_train, y_train)
    rep = report(y_test, model.predict(X_test))
    cv = cross_validate(lambda: make_model(args.model, len(set(y_train.tolist()))),
                        X_train, y_train)
    print(f"dataset={args.dataset or args.csv} model={args.model} "
          f"train={len(X_train)} test={len(X_test)}")
    print(json.dumps({**rep, **cv}, indent=2))


def cmd_compare(args):
    X, y = get_dataset(args.dataset, args.csv, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=args.seed)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    rows = compare_models(X_train, y_train, X_test, y_test)
    print(f"{'model':28s} {'accuracy':>8s} {'f1':>8s}")
    for r in rows:
        print(f"{r['model']:28s} {r['accuracy']:8.4f} {r['f1']:8.4f}")


def main():
    p = argparse.ArgumentParser(description="ML pipeline from scratch")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, func in [("run", cmd_run), ("compare", cmd_compare)]:
        s = sub.add_parser(name)
        s.add_argument("--dataset", default="blobs",
                       choices=["blobs", "moons", "iris", "breast-cancer"])
        s.add_argument("--csv", default=None)
        s.add_argument("--target", default=None)
        s.add_argument("--seed", type=int, default=0)
        if name == "run":
            s.add_argument("--model", default="logreg",
                           choices=["logreg", "knn", "nb"])
        s.set_defaults(func=func)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
