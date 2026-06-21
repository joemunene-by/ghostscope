<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="ghostscope logo">
</p>

<h1 align="center">ghostscope</h1>

<!-- ghostsuite-badges -->
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/joemunene-by/ghostsuite"><img src="https://img.shields.io/badge/part%20of-GhostSuite-6f42c1" alt="Part of GhostSuite"></a>
</p>
<!-- /ghostsuite-badges -->

AI-based network intrusion detection for authorized defensive monitoring.

ghostscope learns a baseline of normal network flow behavior from labeled or
unlabeled flow features, then flags anomalies in new traffic with explainable
alerts that report which features drove each detection. It ships with two
detectors (an IsolationForest and a PyTorch autoencoder), a deterministic
synthetic data generator, and an evaluation harness, so the whole tool runs end
to end with zero external downloads.

## Authorized and defensive use only

ghostscope is a blue-team tool intended for monitoring networks and systems you
own or are explicitly authorized to monitor. Do not use it against systems
without permission. It detects anomalies for incident response and does not
perform any offensive action.

## Install

Requires Python 3.11 or newer.

```
git clone https://github.com/joemunene-by/ghostscope.git
cd ghostscope
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The autoencoder backend needs PyTorch. The `dev` extra installs it. If you only
want the IsolationForest backend, `pip install -e .` is enough and PyTorch is
imported lazily, so the tool still runs without it.

## Quickstart

Generate data, train, detect, and evaluate:

```
ghostscope gen-data --out data/flows.csv
ghostscope train data/flows.csv --model iforest --model-dir models/ghostscope
ghostscope detect data/flows.csv --model-dir models/ghostscope --json-out alerts.json
ghostscope evaluate data/flows.csv --model-dir models/ghostscope
```

Example console output from `detect` (truncated):

```
                          ghostscope alerts
  row     score   top contributing features                       label
 ---------------------------------------------------------------------
    3    0.6312   bytes_out=534118.2, duration=71.4, packets=470       1
   17    0.5980   packets=1421.0, dst_port=80.0, flags=S0              1
   42    0.5571   dst_port=23.0, bytes_out=0.0, duration=0.008         1

168 alert(s) of 2200 records (threshold=0.5402, model=iforest)
```

Example `evaluate` output:

```
        ghostscope evaluation
 precision   0.8810
 recall      0.7400
 f1          0.8043
 roc_auc     0.9775

          confusion matrix
               pred normal   pred anomaly
 true normal          1980             20
 true anomaly           52            148
```

Switch to the autoencoder backend with `--model autoencoder`.

## Feature schema

ghostscope ingests a CSV of flow records. The canonical schema mirrors common
netflow and NSL-KDD style fields.

| Column      | Type        | Description                                  |
|-------------|-------------|----------------------------------------------|
| `duration`  | numeric     | Flow duration in seconds.                    |
| `bytes_in`  | numeric     | Bytes received.                              |
| `bytes_out` | numeric     | Bytes sent.                                  |
| `packets`   | numeric     | Packet count.                                |
| `src_port`  | numeric     | Source port.                                 |
| `dst_port`  | numeric     | Destination port.                            |
| `protocol`  | categorical | Transport protocol (tcp, udp, icmp).         |
| `flags`     | categorical | Connection flag state (SF, S0, REJ, ...).    |
| `label`     | optional    | Ground truth: 0 normal, 1 anomaly.           |

Numeric columns are standard-scaled. Categorical columns are ordinally encoded,
with unseen categories mapped to a reserved unknown value. The `label` column is
optional and is used only by `evaluate`; it is never fed to the detectors, which
train in an unsupervised manner.

A schema mismatch (missing feature column) produces a clear error naming the
missing columns and what was found.

## Models

### IsolationForest (default)

An ensemble of randomized isolation trees. Records that are isolated with few
splits receive high anomaly scores. Fast, no GPU, strong on tabular flow data.
Attribution uses each scaled feature's deviation from the training distribution.

### Autoencoder (PyTorch)

A small fully connected autoencoder trained to reconstruct normal traffic. High
reconstruction error means a record does not resemble the learned baseline.
Attribution uses per-feature reconstruction error, which directly points at the
fields that the model could not reproduce.

Both detectors persist as a bundle directory containing the model, the fitted
feature pipeline, and a metadata JSON (feature names, threshold, model type).

## Threshold calibration

The alert threshold is calibrated from a percentile of the training anomaly
scores (default the 99th percentile). Because training data is dominated by
normal traffic, this sets a sensible cutoff above the bulk of benign scores.
Tune it with `--percentile`, and tune the IsolationForest anomaly fraction with
`--contamination`.

## Metrics

`evaluate` reports:

- precision: fraction of flagged records that are truly anomalous.
- recall: fraction of true anomalies that were flagged.
- f1: harmonic mean of precision and recall.
- roc_auc: ranking quality of the continuous anomaly scores.
- confusion matrix: true and false positives and negatives.

## CLI reference

| Command           | Purpose                                               |
|-------------------|-------------------------------------------------------|
| `gen-data`        | Generate a deterministic synthetic labeled dataset.   |
| `train`           | Train a detector and persist the model bundle.        |
| `detect`          | Score records and emit explainable alerts.            |
| `evaluate`        | Report detection metrics against labeled data.        |
| `info`            | Show tool version, schema, and bundle metadata.       |

Add `--verbose` to any command for debug logging.

## Roadmap

- Streaming detection mode for live flow ingestion.
- Additional backends (one-class SVM, local outlier factor).
- Native NSL-KDD and CIC-IDS column adapters.
- SHAP-based attribution as an opt-in explainer.
- Drift detection to signal when the baseline should be retrained.

## License

MIT. See [LICENSE](LICENSE).

---

<sub>Part of <a href="https://github.com/joemunene-by/ghostsuite">ghostsuite</a>: eleven open-source security tools, one ghost.</sub>
