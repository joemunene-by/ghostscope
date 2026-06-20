"""Detector orchestration: training, scoring, thresholds, explainability, IO.

Wraps the IsolationForest and autoencoder backends behind one interface,
persists the full bundle (model, feature pipeline, threshold, metadata),
and produces explainable per-record attributions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import __version__
from .features import FeaturePipeline
from .logging_utils import get_logger
from .schema import FlowSchema

MODEL_IFOREST = "iforest"
MODEL_AUTOENCODER = "autoencoder"
VALID_MODELS = (MODEL_IFOREST, MODEL_AUTOENCODER)

BUNDLE_MODEL = "model.joblib"
BUNDLE_PIPELINE = "pipeline.joblib"
BUNDLE_META = "metadata.json"


@dataclass
class Metadata:
    """Persisted metadata describing a trained detector bundle."""

    model_type: str
    feature_names: list[str]
    threshold: float
    percentile: float
    contamination: float
    n_train: int
    seed: int
    ghostscope_version: str


class Detector:
    """Unified anomaly detector over a feature pipeline and a backend model.

    Scores follow the convention that higher means more anomalous. A record
    is flagged when its score exceeds ``threshold``.
    """

    def __init__(
        self,
        model_type: str,
        pipeline: FeaturePipeline,
        backend,
        threshold: float,
        metadata: Metadata,
        train_scores: np.ndarray | None = None,
    ) -> None:
        self.model_type = model_type
        self.pipeline = pipeline
        self.backend = backend
        self.threshold = threshold
        self.metadata = metadata
        self.train_scores = train_scores

    # ---- scoring -------------------------------------------------------

    def _raw_scores(self, x: np.ndarray) -> np.ndarray:
        """Anomaly scores where higher means more anomalous."""
        if self.model_type == MODEL_IFOREST:
            # IsolationForest.score_samples: higher means more normal.
            # Negate so higher means more anomalous.
            return -self.backend.score_samples(x)
        return self.backend.score(x)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self.pipeline.transform(df)
        return self._raw_scores(x)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, flags) where flags is 1 for anomalies."""
        scores = self.score(df)
        flags = (scores > self.threshold).astype(int)
        return scores, flags

    # ---- explainability ------------------------------------------------

    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
        """Return per-record top contributing features.

        For the autoencoder, attribution is the per-feature reconstruction
        error. For IsolationForest, attribution is the absolute z-score of
        each (scaled) feature relative to the training distribution, which
        approximates how far each feature deviates from normal.

        Returns:
            One list per input row. Each element is a list of dicts with
            keys ``feature``, ``score``, and ``value`` (the raw input value),
            sorted descending by contribution and truncated to ``top_k``.
        """
        x = self.pipeline.transform(df)
        names = self.pipeline.feature_names or []
        if self.model_type == MODEL_AUTOENCODER:
            contrib = self.backend.per_feature_error(x)
        else:
            # Scaled features already have ~zero mean / unit variance on the
            # training set, so abs(value) is the deviation in std units.
            contrib = np.abs(x)

        raw = self._raw_feature_values(df, names)
        results: list[list[dict]] = []
        for row_idx in range(contrib.shape[0]):
            order = np.argsort(contrib[row_idx])[::-1][:top_k]
            row = [
                {
                    "feature": names[j],
                    "score": float(contrib[row_idx, j]),
                    "value": raw[row_idx][j],
                }
                for j in order
            ]
            results.append(row)
        return results

    def _raw_feature_values(self, df: pd.DataFrame, names: list[str]) -> list[list]:
        """Raw (pre-scaling) values aligned to ``names`` order."""
        values = []
        for _, record in df[names].iterrows():
            row = []
            for name in names:
                val = record[name]
                row.append(val.item() if hasattr(val, "item") else val)
            values.append(row)
        return values

    # ---- persistence ---------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(directory, BUNDLE_PIPELINE))

        if self.model_type == MODEL_IFOREST:
            joblib.dump(self.backend, os.path.join(directory, BUNDLE_MODEL))
        else:
            payload = self.backend.to_payload()
            joblib.dump(payload, os.path.join(directory, BUNDLE_MODEL))

        with open(os.path.join(directory, BUNDLE_META), "w", encoding="utf-8") as fh:
            json.dump(asdict(self.metadata), fh, indent=2)

    @classmethod
    def load(cls, directory: str) -> Detector:
        meta_path = os.path.join(directory, BUNDLE_META)
        if not os.path.isdir(directory) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No ghostscope model bundle found at {directory}. "
                "Run 'ghostscope train' first."
            )
        with open(meta_path, encoding="utf-8") as fh:
            meta = Metadata(**json.load(fh))

        pipeline = joblib.load(os.path.join(directory, BUNDLE_PIPELINE))
        raw = joblib.load(os.path.join(directory, BUNDLE_MODEL))

        if meta.model_type == MODEL_IFOREST:
            backend = raw
        else:
            from .autoencoder import AutoencoderDetector

            backend = AutoencoderDetector.from_payload(raw)

        return cls(
            model_type=meta.model_type,
            pipeline=pipeline,
            backend=backend,
            threshold=meta.threshold,
            metadata=meta,
        )


def _calibrate_threshold(
    scores: np.ndarray, percentile: float
) -> float:
    """Threshold as a percentile of training (mostly normal) anomaly scores."""
    return float(np.percentile(scores, percentile))


def train_detector(
    df: pd.DataFrame,
    model_type: str = MODEL_IFOREST,
    schema: FlowSchema | None = None,
    contamination: float = 0.02,
    percentile: float = 99.0,
    seed: int = 1337,
    epochs: int = 40,
) -> Detector:
    """Train a detector on flow records.

    Training is unsupervised: any ``label`` column is ignored for fitting.
    The threshold is calibrated from a percentile of the training scores.

    Args:
        df: training flow records.
        model_type: 'iforest' or 'autoencoder'.
        schema: feature schema (defaults to canonical).
        contamination: expected anomaly fraction for IsolationForest.
        percentile: percentile of training scores used as the threshold.
        seed: global seed for determinism.
        epochs: autoencoder training epochs (ignored for iforest).
    """
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose one of {VALID_MODELS}."
        )
    logger = get_logger()
    schema = schema or FlowSchema()

    pipeline = FeaturePipeline(schema=schema)
    x = pipeline.fit_transform(df)
    logger.debug("Fitted feature pipeline: %d features", x.shape[1])

    if model_type == MODEL_IFOREST:
        backend = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=seed,
        )
        backend.fit(x)
        train_scores = -backend.score_samples(x)
    else:
        from .autoencoder import AutoencoderDetector

        backend = AutoencoderDetector(epochs=epochs, seed=seed)
        backend.fit(x)
        train_scores = backend.score(x)

    # Calibrate the threshold from the normal baseline. When a label column is
    # present we use the scores of known-normal records (held-out normal set
    # semantics), so the percentile cutoff sits just above benign traffic and
    # genuine anomalies, which score higher, are flagged. With unlabeled data
    # we fall back to all training scores.
    labels = FeaturePipeline.split_labels(df, schema)
    if labels is not None and (labels == 0).any():
        calib_scores = train_scores[labels == 0]
    else:
        calib_scores = train_scores
    threshold = _calibrate_threshold(calib_scores, percentile)
    logger.debug("Calibrated threshold at p%.1f = %.6f", percentile, threshold)

    metadata = Metadata(
        model_type=model_type,
        feature_names=pipeline.feature_names or [],
        threshold=threshold,
        percentile=percentile,
        contamination=contamination,
        n_train=len(df),
        seed=seed,
        ghostscope_version=__version__,
    )
    return Detector(
        model_type=model_type,
        pipeline=pipeline,
        backend=backend,
        threshold=threshold,
        metadata=metadata,
        train_scores=train_scores,
    )
