# Changelog

All notable changes to ghostscope are documented in this file. The format is
based on Keep a Changelog, and this project adheres to semantic versioning.

## [0.1.0] - 2026-06-20

### Added

- Initial release of ghostscope, an AI-based network intrusion detection tool
  for authorized defensive monitoring.
- Two anomaly-detection backends: scikit-learn IsolationForest (default) and a
  PyTorch reconstruction-error autoencoder, selectable with `--model`.
- Feature pipeline with ordinal encoding for categorical fields and standard
  scaling for numerics. The encoder and scaler are persisted with the model.
- Deterministic synthetic data generator (`gen-data`) producing normal traffic
  clusters plus injected port-scan, DDoS-burst, and exfiltration anomalies.
- Explainable alerts: top contributing features per flagged record, using
  per-feature reconstruction error (autoencoder) or scaled deviation
  (IsolationForest).
- Threshold calibration from a configurable percentile of training scores.
- `evaluate` reports precision, recall, F1, ROC-AUC, and a confusion matrix.
- Typer CLI with `gen-data`, `train`, `detect`, `evaluate`, and `info`.
- Deterministic, offline pytest suite and a ruff + pytest CI workflow.
