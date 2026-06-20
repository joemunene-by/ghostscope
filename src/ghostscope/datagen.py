"""Deterministic synthetic flow data generator.

Produces a labeled CSV combining normal traffic clusters with injected
anomalies (port scans, DDoS bursts, data exfiltration). Everything is
seeded so a given seed always yields an identical dataset, which keeps the
whole tool runnable with zero external downloads and makes tests
reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURES

PROTOCOLS = ["tcp", "udp", "icmp"]
FLAG_STATES = ["SF", "S0", "REJ", "RSTO", "OTH"]


def _normal_traffic(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate normal traffic drawn from a few well-behaved clusters."""
    # Three benign service clusters: web, dns, internal-bulk.
    cluster = rng.integers(0, 3, size=n)
    duration = np.empty(n)
    bytes_in = np.empty(n)
    bytes_out = np.empty(n)
    packets = np.empty(n)
    src_port = np.empty(n, dtype=int)
    dst_port = np.empty(n, dtype=int)
    protocol = np.empty(n, dtype=object)
    flags = np.empty(n, dtype=object)

    for idx in range(n):
        c = cluster[idx]
        if c == 0:  # web (https)
            duration[idx] = abs(rng.normal(2.0, 0.8))
            bytes_in[idx] = abs(rng.normal(1500, 400))
            bytes_out[idx] = abs(rng.normal(800, 250))
            packets[idx] = abs(rng.normal(14, 4)) + 1
            src_port[idx] = rng.integers(1024, 65535)
            dst_port[idx] = 443
            protocol[idx] = "tcp"
            flags[idx] = "SF"
        elif c == 1:  # dns
            duration[idx] = abs(rng.normal(0.2, 0.1))
            bytes_in[idx] = abs(rng.normal(120, 40))
            bytes_out[idx] = abs(rng.normal(90, 30))
            packets[idx] = abs(rng.normal(2, 1)) + 1
            src_port[idx] = rng.integers(1024, 65535)
            dst_port[idx] = 53
            protocol[idx] = "udp"
            flags[idx] = "SF"
        else:  # internal bulk transfer
            duration[idx] = abs(rng.normal(5.0, 1.5))
            bytes_in[idx] = abs(rng.normal(4000, 1000))
            bytes_out[idx] = abs(rng.normal(3500, 900))
            packets[idx] = abs(rng.normal(40, 10)) + 1
            src_port[idx] = rng.integers(1024, 65535)
            dst_port[idx] = rng.choice([22, 445, 3306])
            protocol[idx] = "tcp"
            flags[idx] = "SF"

    return pd.DataFrame(
        {
            "duration": duration,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "packets": packets,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "flags": flags,
            LABEL_COLUMN: np.zeros(n, dtype=int),
        }
    )


def _port_scan(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Port scan: tiny flows, many distinct dst ports, half-open connections."""
    return pd.DataFrame(
        {
            "duration": abs(rng.normal(0.01, 0.005, n)),
            "bytes_in": abs(rng.normal(40, 10, n)),
            "bytes_out": abs(rng.normal(0, 5, n)),
            "packets": np.ones(n) + rng.integers(0, 2, n),
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": rng.integers(1, 1024, n),
            "protocol": rng.choice(["tcp"], n),
            "flags": rng.choice(["S0", "REJ"], n),
            LABEL_COLUMN: np.ones(n, dtype=int),
        }
    )


def _ddos_burst(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """DDoS burst: huge packet counts, short duration, hammering one port."""
    return pd.DataFrame(
        {
            "duration": abs(rng.normal(0.05, 0.02, n)),
            "bytes_in": abs(rng.normal(60, 20, n)),
            "bytes_out": abs(rng.normal(20, 10, n)),
            "packets": abs(rng.normal(1200, 300, n)) + 200,
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": rng.choice([80, 443], n),
            "protocol": rng.choice(["tcp", "udp"], n),
            "flags": rng.choice(["S0", "OTH"], n),
            LABEL_COLUMN: np.ones(n, dtype=int),
        }
    )


def _exfil(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Data exfiltration: long flows, massive outbound bytes, odd port."""
    return pd.DataFrame(
        {
            "duration": abs(rng.normal(60, 15, n)) + 10,
            "bytes_in": abs(rng.normal(500, 150, n)),
            "bytes_out": abs(rng.normal(500000, 120000, n)) + 100000,
            "packets": abs(rng.normal(400, 80, n)) + 50,
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": rng.choice([8080, 4444, 31337], n),
            "protocol": rng.choice(["tcp"], n),
            "flags": rng.choice(["SF", "RSTO"], n),
            LABEL_COLUMN: np.ones(n, dtype=int),
        }
    )


def generate_dataset(
    n_normal: int = 2000,
    n_port_scan: int = 80,
    n_ddos: int = 80,
    n_exfil: int = 40,
    seed: int = 1337,
) -> pd.DataFrame:
    """Generate a deterministic labeled flow dataset.

    Args:
        n_normal: count of benign flow records.
        n_port_scan: count of injected port-scan anomalies.
        n_ddos: count of injected DDoS-burst anomalies.
        n_exfil: count of injected exfiltration anomalies.
        seed: RNG seed; identical seed yields an identical dataset.

    Returns:
        A shuffled DataFrame with the canonical feature columns plus a
        ``label`` column (0 = normal, 1 = anomaly).
    """
    rng = np.random.default_rng(seed)
    frames = [
        _normal_traffic(rng, n_normal),
        _port_scan(rng, n_port_scan),
        _ddos_burst(rng, n_ddos),
        _exfil(rng, n_exfil),
    ]
    data = pd.concat(frames, ignore_index=True)

    # Deterministic shuffle.
    order = rng.permutation(len(data))
    data = data.iloc[order].reset_index(drop=True)

    # Enforce dtypes and column order.
    for col in NUMERIC_FEATURES:
        data[col] = data[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(str)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(int)

    return data[FEATURE_COLUMNS + [LABEL_COLUMN]]


def write_dataset(path: str, **kwargs) -> pd.DataFrame:
    """Generate a dataset and write it to ``path`` as CSV. Returns the frame."""
    data = generate_dataset(**kwargs)
    data.to_csv(path, index=False)
    return data
