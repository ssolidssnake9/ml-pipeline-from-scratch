# ML Pipeline from Scratch

Classifiers and evaluation metrics implemented from scratch in NumPy —
no `sklearn` under the hood for the learning itself — wrapped in a real
pipeline (split → scale → train → cross-validate → evaluate) with a CLI
that benchmarks the from-scratch models against their sklearn twins.

## What's from scratch

- **Logistic regression** — vectorized gradient descent, L2 option
- **k-NN** — broadcast distance matrix, majority vote
- **Gaussian Naive Bayes** — log-space for numerical stability
- **Metrics** — accuracy, precision, recall, F1 (macro/micro),
  confusion matrix
- **Pipeline plumbing** — seeded train/test split, standard scaler,
  one-vs-rest wrapper, k-fold cross-validation

sklearn is used only for the *comparison* baseline and built-in datasets.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train + evaluate one model
python src/cli.py run --dataset blobs --model logreg

# from-scratch vs sklearn, side by side
python src/cli.py compare --dataset breast-cancer
```

Datasets: `blobs`, `moons` (synthetic), `iris`, `breast-cancer`
(sklearn built-ins), or `--csv data.csv --target label` for your own.

## Example output

`compare --dataset breast-cancer` prints per-model accuracy and F1 for
each from-scratch implementation next to its sklearn counterpart — the
whole point is watching the NumPy versions land within a whisker of the
battle-tested ones.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Layout

```
src/
  classifiers.py  # LogisticRegression, KNN, GaussianNB (NumPy only)
  metrics.py      # accuracy, precision/recall/F1, confusion matrix
  pipeline.py     # split, scaler, one-vs-rest, CV, evaluate, compare
  datasets.py     # synthetic + sklearn + CSV loaders
  cli.py          # run / compare commands
tests/
```
