"""
raretect — rare-event detection toolkit for extreme imbalanced scientific tabular data.
"""

from raretect.data import generate_synthetic_detector_data
from raretect.pipeline import Pipeline
from raretect.metrics import print_rare_event_report, pr_auc_score, roc_auc_score, precision_at_recall

__version__ = "0.1.0"
__all__ = [
    "Pipeline",
    "generate_synthetic_detector_data",
    "print_rare_event_report",
    "pr_auc_score",
    "roc_auc_score",
    "precision_at_recall",
]
