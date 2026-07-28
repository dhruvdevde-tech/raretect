"""
Metrics and evaluation report functions tuned for extreme imbalance (1% signal).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


def pr_auc_score(y_true, y_scores) -> float:
    """Calculate Precision-Recall Area Under the Curve (PR-AUC)."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return float(auc(recall, precision))


def precision_at_recall(y_true, y_scores, target_recall: float = 0.90) -> float:
    """Calculate precision at a specified target recall threshold."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(precision[idx]))


def print_rare_event_report(y_true, y_scores, target_recall: float = 0.90) -> dict:
    """
    Generate evaluation report metrics suitable for rare event detection.

    Returns
    -------
    metrics : dict
        Dictionary containing 'pr_auc', 'roc_auc', and 'precision_at_recall'.
    """
    pr = pr_auc_score(y_true, y_scores)
    roc = float(roc_auc_score(y_true, y_scores))
    prec_at_rec = precision_at_recall(y_true, y_scores, target_recall=target_recall)

    top_n = int(np.sum(y_true))
    top_indices = np.argsort(y_scores)[::-1][:top_n]
    top_hits = int(np.sum(y_true[top_indices]))

    return {
        "pr_auc": round(pr, 4),
        "roc_auc": round(roc, 4),
        f"precision_at_{int(target_recall*100)}rec": round(prec_at_rec, 4),
        "top_hits": f"{top_hits}/{top_n}",
    }
