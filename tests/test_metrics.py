"""Tests for evaluation metric computation on known cases."""

from __future__ import annotations

import numpy as np

from ghostscope.metrics import evaluate


def test_known_confusion_case():
    # 4 normal, 4 anomaly. Predictions: 1 FP, 1 FN.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    report = evaluate(y_true, y_pred)
    assert report.true_negatives == 3
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.true_positives == 3
    # precision = 3 / (3 + 1) = 0.75; recall = 3 / (3 + 1) = 0.75
    assert abs(report.precision - 0.75) < 1e-9
    assert abs(report.recall - 0.75) < 1e-9
    assert abs(report.f1 - 0.75) < 1e-9
    assert report.support_normal == 4
    assert report.support_anomaly == 4


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = evaluate(y_true, y_pred)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


def test_roc_auc_from_scores():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    report = evaluate(y_true, y_pred, scores)
    assert report.roc_auc == 1.0


def test_roc_auc_none_when_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    report = evaluate(y_true, y_pred, scores)
    assert report.roc_auc is None
