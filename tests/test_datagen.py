"""Tests for the deterministic synthetic data generator."""

from __future__ import annotations

import pandas as pd

from ghostscope.datagen import generate_dataset, write_dataset
from ghostscope.schema import FEATURE_COLUMNS, LABEL_COLUMN


def test_generate_is_deterministic():
    a = generate_dataset(n_normal=200, seed=42)
    b = generate_dataset(n_normal=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generate_differs_by_seed():
    a = generate_dataset(n_normal=200, seed=1)
    b = generate_dataset(n_normal=200, seed=2)
    assert not a.equals(b)


def test_columns_and_label_counts():
    df = generate_dataset(
        n_normal=500, n_port_scan=10, n_ddos=10, n_exfil=5, seed=7
    )
    assert list(df.columns) == FEATURE_COLUMNS + [LABEL_COLUMN]
    assert len(df) == 525
    assert int(df[LABEL_COLUMN].sum()) == 25
    assert df[LABEL_COLUMN].isin([0, 1]).all()


def test_write_dataset_roundtrip(tmp_path):
    out = tmp_path / "flows.csv"
    df = write_dataset(str(out), n_normal=100, seed=3)
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)
