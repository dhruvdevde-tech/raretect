"""
Model implementations and rank-average ensemble logic for rare-event scoring.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    """Wrapper around scikit-learn IsolationForest for rare-event detection."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X) -> np.ndarray:
        # Decision function returns higher values for inliers, invert so higher = more anomalous
        return -self.model.decision_function(X)


class RobustMahalanobisModel:
    """Statistical multi-channel deviation model using robust covariance."""

    def __init__(self):
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X):
        X_arr = np.asarray(X)
        self.mean_ = np.median(X_arr, axis=0)
        cov = np.cov(X_arr, rowvar=False) + np.eye(X_arr.shape[1]) * 1e-4
        self.inv_cov_ = np.linalg.pinv(cov)
        return self

    def score(self, X) -> np.ndarray:
        X_arr = np.asarray(X)
        diff = X_arr - self.mean_
        # Mahalanobis distance squared
        scores = np.sum(np.dot(diff, self.inv_cov_) * diff, axis=1)
        return np.sqrt(np.maximum(0, scores))


class RankAverageEnsemble:
    """Ensemble model that normalizes scores using rank averaging to prevent single model dominance."""

    def __init__(self, models=None):
        if models is None:
            models = [IsolationForestModel(), RobustMahalanobisModel()]
        self.models = models

    def fit(self, X):
        for m in self.models:
            m.fit(X)
        return self

    def score(self, X) -> np.ndarray:
        n = len(X)
        rank_sum = np.zeros(n, dtype=float)
        for m in self.models:
            raw_scores = m.score(X)
            # Convert raw anomaly scores into percentile ranks [0, 1]
            ranks = rankdata(raw_scores) / n
            rank_sum += ranks
        return rank_sum / len(self.models)
