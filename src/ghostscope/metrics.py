"""Evaluation metrics for labeled detection results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvalReport:
    """Container for evaluation metrics on a labeled dataset."""

    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    support_normal: int
    support_anomaly: int

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": {
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_positives": self.true_positives,
            },
            "support": {
                "normal": self.support_normal,
                "anomaly": self.support_anomaly,
            },
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> EvalReport:
    """Compute precision/recall/F1/ROC-AUC and a confusion matrix.

    Args:
        y_true: ground-truth labels (1 = anomaly, 0 = normal).
        y_pred: predicted flags (1 = anomaly).
        scores: optional continuous anomaly scores for ROC-AUC.

    Returns:
        An EvalReport. ROC-AUC is None when scores are absent or only one
        class is present in y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, pos_label=1
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc_auc: float | None = None
    if scores is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))

    return EvalReport(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        support_normal=int((y_true == 0).sum()),
        support_anomaly=int((y_true == 1).sum()),
    )
