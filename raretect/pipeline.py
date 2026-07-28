"""
Main 4-stage Pipeline class for raretect detection workflows.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from raretect.models import RankAverageEnsemble
from raretect.metrics import print_rare_event_report


class Pipeline:
    """
    4-stage Raretect Pipeline for rare event detection.

    Stage 1: Feature Engineering (Row-wise stats & z-score deviations)
    Stage 2: Robust Preprocessing (Median imputation & RobustScaler)
    Stage 3: Rank-averaged Ensemble Modeling
    Stage 4: Imbalanced Evaluation (PR-AUC, ROC-AUC, Precision @ Recall)
    """

    def __init__(self, contamination: float = 0.01, models=None):
        self.contamination = contamination
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.ensemble = RankAverageEnsemble(models=models)
        self.is_fitted = False

    def _engineer_features(self, X):
        X_arr = np.asarray(X)
        row_means = np.mean(X_arr, axis=1, keepdims=True)
        row_stds = np.std(X_arr, axis=1, keepdims=True) + 1e-6
        row_maxs = np.max(X_arr, axis=1, keepdims=True)
        z_deviations = (X_arr - row_means) / row_stds

        # Concatenate engineered features
        engineered = np.hstack([X_arr, row_means, row_stds, row_maxs, z_deviations])
        return engineered

    def fit(self, X):
        """Fit preprocessing transformers and ensemble models on input detector data."""
        engineered = self._engineer_features(X)
        imputed = self.imputer.fit_transform(engineered)
        scaled = self.scaler.fit_transform(imputed)
        self.ensemble.fit(scaled)
        self.is_fitted = True
        return self

    def score(self, X) -> np.ndarray:
        """Calculate rare-event anomaly scores (higher score = more anomalous)."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before scoring.")
        engineered = self._engineer_features(X)
        imputed = self.imputer.transform(engineered)
        scaled = self.scaler.transform(imputed)
        return self.ensemble.score(scaled)

    def predict(self, X, threshold: float = 0.95) -> np.ndarray:
        """Return binary prediction labels (1 = rare event, 0 = background)."""
        scores = self.score(X)
        cutoff = np.quantile(scores, threshold)
        return (scores >= cutoff).astype(int)

    def evaluate(self, X, y_true, target_recall: float = 0.90) -> dict:
        """Evaluate rare-event detection metrics on test data."""
        scores = self.score(X)
        return print_rare_event_report(y_true, scores, target_recall=target_recall)
