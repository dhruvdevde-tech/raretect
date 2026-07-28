"""
Unit test suite verifying all 9 core raretect pipeline capabilities.
"""

import numpy as np
import pandas as pd
import pytest
from raretect import Pipeline, generate_synthetic_detector_data
from raretect.metrics import pr_auc_score, roc_auc_score, precision_at_recall


def test_01_data_generator_shape():
    df, labels = generate_synthetic_detector_data(n_samples=1000, n_features=15, contamination=0.01)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 15)
    assert len(labels) == 1000
    assert np.sum(labels) == 10


def test_02_data_generator_reproducibility():
    df1, l1 = generate_synthetic_detector_data(n_samples=500, random_state=42)
    df2, l2 = generate_synthetic_detector_data(n_samples=500, random_state=42)
    np.testing.assert_array_equal(df1.values, df2.values)
    np.testing.assert_array_equal(l1, l2)


def test_03_pipeline_fit_predict():
    df, labels = generate_synthetic_detector_data(n_samples=1000, n_features=10, contamination=0.02)
    pipe = Pipeline(contamination=0.02)
    pipe.fit(df)
    assert pipe.is_fitted
    scores = pipe.score(df)
    assert len(scores) == 1000
    assert np.all(scores >= 0) and np.all(scores <= 1.0)


def test_04_pipeline_predict_threshold():
    df, _ = generate_synthetic_detector_data(n_samples=1000, n_features=10)
    pipe = Pipeline().fit(df)
    preds = pipe.predict(df, threshold=0.95)
    assert len(preds) == 1000
    # Top 5% of events predicted as 1
    assert np.sum(preds) == pytest.approx(50, abs=5)


def test_05_unfitted_pipeline_raises_error():
    df, _ = generate_synthetic_detector_data(n_samples=100)
    pipe = Pipeline()
    with pytest.raises(RuntimeError):
        pipe.score(df)


def test_06_pr_auc_score_calculation():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.1, 0.3, 0.4, 0.9, 0.85, 0.2, 0.15, 0.1])
    score = pr_auc_score(y_true, y_scores)
    assert score > 0.7


def test_07_roc_auc_score_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.85])
    roc = roc_auc_score(y_true, y_scores)
    assert roc == 1.0


def test_08_precision_at_recall_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.85])
    prec = precision_at_recall(y_true, y_scores, target_recall=0.90)
    assert prec == 1.0


def test_09_full_pipeline_evaluate():
    df, labels = generate_synthetic_detector_data(n_samples=2000, n_features=20, contamination=0.01)
    pipe = Pipeline(contamination=0.01)
    pipe.fit(df)
    metrics = pipe.evaluate(df, labels, target_recall=0.90)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.85
    assert metrics["pr_auc"] > 0.40
