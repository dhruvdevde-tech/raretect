# raretect — rare-event detection toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)

**raretect** is a Python toolkit built for research labs where target signals are 1% of the data — dark matter events, rare particle collisions, spectroscopy anomalies, and industrial sensor monitoring.

---

## ⚡ Quickstart

Install `raretect` directly from source:

```bash
pip install -e .
```

Run a 5-line pipeline on synthetic detector data:

```python
from raretect import Pipeline, generate_synthetic_detector_data

# 1. Generate synthetic 20-channel detector data with 1% signal contamination
df, labels = generate_synthetic_detector_data(n_samples=5000, n_features=20, contamination=0.01)

# 2. Initialize 4-stage pipeline
pipe = Pipeline(contamination=0.01)

# 3. Fit pipeline on detector data
pipe.fit(df)

# 4. Score events (higher score = more anomalous)
scores = pipe.score(df)

# 5. Evaluate imbalanced metrics (PR-AUC, ROC-AUC, Precision @ Recall)
metrics = pipe.evaluate(df, labels)
print(metrics)
# Output: {'pr_auc': 0.77, 'roc_auc': 0.997, 'precision_at_90rec': 0.55, 'top_hits': '50/50'}
```

---

## 🔬 How it Works

`raretect` implements a 4-stage architecture tuned for sparse, high-dimensional, imbalanced data:

1. **Feature Engineering**: Row-wise statistics and per-channel z-score deviations so multi-channel signals don't hide in average feature metrics.
2. **Robust Preprocessing**: Median imputation and `RobustScaler` designed for outlier-heavy detector backgrounds.
3. **Rank-Averaged Ensemble**: Isolation Forest and Robust Mahalanobis distance models combined via percentile rank-averaging to eliminate single-model blind spots.
4. **Imbalanced Metrics Evaluation**: PR-AUC, ROC-AUC, and precision at target recall thresholds — metric standards built for 1% contamination.

---

## 🧪 Testing

Run unit tests with `pytest`:

```bash
pytest -v
```

Output:
```text
tests/test_pipeline.py::test_01_data_generator_shape PASSED
tests/test_pipeline.py::test_02_data_generator_reproducibility PASSED
tests/test_pipeline.py::test_03_pipeline_fit_predict PASSED
tests/test_pipeline.py::test_04_pipeline_predict_threshold PASSED
tests/test_pipeline.py::test_05_unfitted_pipeline_raises_error PASSED
tests/test_pipeline.py::test_06_pr_auc_score_calculation PASSED
tests/test_pipeline.py::test_07_roc_auc_score_calculation PASSED
tests/test_pipeline.py::test_08_precision_at_recall_calculation PASSED
tests/test_pipeline.py::test_09_full_pipeline_evaluate PASSED

9 passed in 2.56s
```

---

## 📄 License
MIT License
