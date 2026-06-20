"""Feature pipeline: encode categoricals, scale numerics, persistable.

The pipeline fits an ordinal encoder over categorical columns and a
standard scaler over the full numeric-plus-encoded matrix. It is
serializable so that train and detect use exactly the same transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .schema import FlowSchema, SchemaError


@dataclass
class FeaturePipeline:
    """Fits and applies the encode + scale transform for flow features."""

    schema: FlowSchema
    encoder: OrdinalEncoder | None = None
    scaler: StandardScaler | None = None
    feature_names: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        """Fit encoder and scaler on a training DataFrame."""
        self.schema.validate(list(df.columns))
        cat = df[self.schema.categorical].astype(str)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.encoder.fit(cat)

        matrix = self._assemble(df)
        self.scaler = StandardScaler()
        self.scaler.fit(matrix)
        self.feature_names = self.schema.numeric + self.schema.categorical
        return self

    def _assemble(self, df: pd.DataFrame) -> np.ndarray:
        """Build the raw (unscaled) feature matrix: numerics + encoded cats."""
        if self.encoder is None:
            raise RuntimeError("FeaturePipeline must be fit before use.")
        num = df[self.schema.numeric].to_numpy(dtype=float)
        cat = self.encoder.transform(df[self.schema.categorical].astype(str))
        return np.hstack([num, cat])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into a scaled feature matrix."""
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("FeaturePipeline must be fit before transform.")
        self.schema.validate(list(df.columns))
        matrix = self._assemble(df)
        return self.scaler.transform(matrix)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @staticmethod
    def split_labels(df: pd.DataFrame, schema: FlowSchema) -> np.ndarray | None:
        """Return the label vector if present, else None."""
        if schema.label in df.columns:
            return df[schema.label].to_numpy(dtype=int)
        return None

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        """Read a flow CSV, raising a SchemaError on an empty file."""
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise SchemaError(f"Input CSV {path} is empty.") from exc
        if df.empty:
            raise SchemaError(f"Input CSV {path} has no rows.")
        return df
