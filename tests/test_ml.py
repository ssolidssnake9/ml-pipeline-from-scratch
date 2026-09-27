"""Tests: metrics, from-scratch classifiers, and the pipeline on tiny data."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classifiers import GaussianNB, KNN
from logreg import LogisticRegression
from metrics import accuracy, confusion_matrix, precision_recall_f1
from pipeline import StandardScaler, cross_validate, train_test_split


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)

    def test_confusion_matrix(self):
        cm, classes = confusion_matrix(np.array([0, 1, 1]), np.array([0, 0, 1]))
        self.assertEqual(cm.tolist(), [[1, 0], [1, 1]])
        self.assertEqual(classes.tolist(), [0, 1])

    def test_prf(self):
        out = precision_recall_f1(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]),
                                  average="micro")
        self.assertAlmostEqual(out["precision"], 0.75)


class TestClassifiers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([rng.normal(-2, 0.5, (30, 2)),
                            rng.normal(2, 0.5, (30, 2))])
        self.y = np.repeat([0, 1], 30)

    def test_logreg_separable(self):
        m = LogisticRegression(lr=0.5, epochs=500).fit(self.X, self.y)
        self.assertGreater(accuracy(self.y, m.predict(self.X)), 0.95)

    def test_knn(self):
        m = KNN(k=3).fit(self.X, self.y)
        self.assertGreater(accuracy(self.y, m.predict(self.X)), 0.95)

    def test_nb(self):
        m = GaussianNB().fit(self.X, self.y)
        self.assertGreater(accuracy(self.y, m.predict(self.X)), 0.95)


class TestPipeline(unittest.TestCase):
    def test_split_sizes(self):
        X = np.zeros((100, 2))
        y = np.zeros(100)
        tr_x, te_x, _, _ = train_test_split(X, y, test_size=0.2, seed=1)
        self.assertEqual(len(tr_x), 80)
        self.assertEqual(len(te_x), 20)

    def test_scaler(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xs = StandardScaler().fit_transform(X)
        self.assertAlmostEqual(Xs.mean(), 0.0, places=9)

    def test_cv(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))])
        y = np.repeat([0, 1], 30)
        out = cross_validate(lambda: KNN(k=3), X, y, k=3)
        self.assertGreater(out["cv_accuracy_mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
