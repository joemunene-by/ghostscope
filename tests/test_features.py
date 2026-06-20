"""Tests for the feature pipeline and schema validation."""

from __future__ import annotations

import numpy as np
import pytest

from ghostscope.datagen import generate_dataset
from ghostscope.features import FeaturePipeline
from ghostscope.schema import FlowSchema, SchemaError


def test_fit_transform_shape():
    df = generate_dataset(n_normal=100, seed=11)
    pipe = FeaturePipeline(schema=FlowSchema())
    x = pipe.fit_transform(df)
    assert x.shape[0] == len(df)
    # 6 numeric + 2 categorical = 8 features
    assert x.shape[1] == 8
    assert pipe.feature_names == [
        "duration",
        "bytes_in",
        "bytes_out",
        "packets",
        "src_port",
        "dst_port",
        "protocol",
        "flags",
    ]


def test_scaling_is_standardized():
    # Use a mixed dataset so every feature column (including categoricals) has
    # variance; a constant column scales to zero std, which is expected.
    df = generate_dataset(
        n_normal=400, n_port_scan=20, n_ddos=20, n_exfil=10, seed=5
    )
    pipe = FeaturePipeline(schema=FlowSchema())
    x = pipe.fit_transform(df)
    # Standardized training features should have near-zero mean, unit std.
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-6)


def test_transform_consistency():
    df = generate_dataset(n_normal=100, seed=13)
    pipe = FeaturePipeline(schema=FlowSchema())
    pipe.fit(df)
    a = pipe.transform(df)
    b = pipe.transform(df)
    assert np.array_equal(a, b)


def test_schema_missing_column_raises():
    df = generate_dataset(n_normal=50, seed=1).drop(columns=["bytes_in"])
    pipe = FeaturePipeline(schema=FlowSchema())
    with pytest.raises(SchemaError, match="bytes_in"):
        pipe.fit(df)


def test_unknown_category_handled():
    df = generate_dataset(n_normal=100, seed=2)
    pipe = FeaturePipeline(schema=FlowSchema())
    pipe.fit(df)
    novel = df.iloc[:5].copy()
    novel["protocol"] = "quic"  # unseen category
    x = pipe.transform(novel)
    assert x.shape == (5, 8)


def test_read_csv_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(SchemaError):
        FeaturePipeline.read_csv(str(empty))
