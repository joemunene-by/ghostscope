"""Tests for detector training, detection, persistence, and explainability."""

from __future__ import annotations

import numpy as np
import pytest

from ghostscope.datagen import generate_dataset
from ghostscope.detector import (
    MODEL_AUTOENCODER,
    MODEL_IFOREST,
    Detector,
    train_detector,
)
from ghostscope.metrics import evaluate
from ghostscope.schema import LABEL_COLUMN


def _dataset():
    return generate_dataset(
        n_normal=1500, n_port_scan=60, n_ddos=60, n_exfil=30, seed=1337
    )


def test_iforest_train_detect_recall():
    df = _dataset()
    detector = train_detector(df, model_type=MODEL_IFOREST, percentile=98.0, seed=1337)
    scores, flags = detector.predict(df)
    report = evaluate(df[LABEL_COLUMN].to_numpy(), flags, scores)
    assert report.recall >= 0.7
    assert report.roc_auc is not None and report.roc_auc >= 0.9


def test_autoencoder_train_detect_recall():
    pytest.importorskip("torch")
    df = _dataset()
    detector = train_detector(
        df, model_type=MODEL_AUTOENCODER, percentile=97.0, seed=1337, epochs=30
    )
    scores, flags = detector.predict(df)
    report = evaluate(df[LABEL_COLUMN].to_numpy(), flags, scores)
    assert report.recall >= 0.7
    assert report.roc_auc is not None and report.roc_auc >= 0.85


def test_iforest_persistence_roundtrip(tmp_path):
    df = _dataset()
    detector = train_detector(df, model_type=MODEL_IFOREST, seed=1337)
    scores_before, flags_before = detector.predict(df)
    detector.save(str(tmp_path / "bundle"))

    loaded = Detector.load(str(tmp_path / "bundle"))
    scores_after, flags_after = loaded.predict(df)
    assert np.allclose(scores_before, scores_after)
    assert np.array_equal(flags_before, flags_after)
    assert loaded.metadata.model_type == MODEL_IFOREST


def test_autoencoder_persistence_roundtrip(tmp_path):
    pytest.importorskip("torch")
    df = _dataset()
    detector = train_detector(
        df, model_type=MODEL_AUTOENCODER, seed=1337, epochs=20
    )
    scores_before, _ = detector.predict(df)
    detector.save(str(tmp_path / "ae"))

    loaded = Detector.load(str(tmp_path / "ae"))
    scores_after, _ = loaded.predict(df)
    assert np.allclose(scores_before, scores_after, atol=1e-5)
    assert loaded.metadata.model_type == MODEL_AUTOENCODER


def test_explain_returns_top_k():
    df = _dataset()
    detector = train_detector(df, model_type=MODEL_IFOREST, seed=1337)
    sample = df.iloc[:10]
    explanations = detector.explain(sample, top_k=3)
    assert len(explanations) == 10
    for row in explanations:
        assert len(row) == 3
        for contrib in row:
            assert set(contrib.keys()) == {"feature", "score", "value"}
        # contributions sorted descending
        scores = [c["score"] for c in row]
        assert scores == sorted(scores, reverse=True)


def test_explain_autoencoder_top_k():
    pytest.importorskip("torch")
    df = _dataset()
    detector = train_detector(
        df, model_type=MODEL_AUTOENCODER, seed=1337, epochs=15
    )
    explanations = detector.explain(df.iloc[:5], top_k=2)
    assert len(explanations) == 5
    assert all(len(row) == 2 for row in explanations)


def test_unknown_model_raises():
    df = _dataset()
    with pytest.raises(ValueError, match="Unknown model_type"):
        train_detector(df, model_type="bogus")


def test_load_missing_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector.load(str(tmp_path / "nope"))


def test_training_is_deterministic():
    df = _dataset()
    d1 = train_detector(df, model_type=MODEL_IFOREST, seed=1337)
    d2 = train_detector(df, model_type=MODEL_IFOREST, seed=1337)
    s1, _ = d1.predict(df)
    s2, _ = d2.predict(df)
    assert np.allclose(s1, s2)
