"""Flow record schema definitions and validation.

ghostscope ingests CSV flow records. The canonical schema mirrors common
netflow / NSL-KDD style fields. Numeric columns are scaled, categorical
columns are ordinally encoded. An optional ``label`` column (0 = normal,
1 = anomaly) is used only by ``evaluate`` and never fed to the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical numeric feature columns expected in a flow CSV.
NUMERIC_FEATURES: list[str] = [
    "duration",
    "bytes_in",
    "bytes_out",
    "packets",
    "src_port",
    "dst_port",
]

# Canonical categorical feature columns. These are ordinally encoded.
CATEGORICAL_FEATURES: list[str] = [
    "protocol",
    "flags",
]

# Optional ground-truth column. Excluded from all feature math.
LABEL_COLUMN: str = "label"

# All feature columns the model consumes (before encoding expansion).
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SchemaError(ValueError):
    """Raised when an input CSV does not satisfy the expected schema."""


@dataclass
class FlowSchema:
    """Describes which columns are numeric vs categorical for a dataset.

    Defaults to the canonical netflow schema but can be customized so the
    tool works with generic numeric/categorical flow exports too.
    """

    numeric: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    label: str = LABEL_COLUMN

    @property
    def feature_columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def validate(self, columns: list[str]) -> None:
        """Validate that all required feature columns are present.

        Raises:
            SchemaError: if any required feature column is missing, listing
                the missing columns and what was actually found.
        """
        present = set(columns)
        required = self.feature_columns
        missing = [c for c in required if c not in present]
        if missing:
            raise SchemaError(
                "Input CSV is missing required feature column(s): "
                f"{', '.join(missing)}. "
                f"Expected feature columns: {', '.join(required)}. "
                f"Found columns: {', '.join(columns)}."
            )

    def to_dict(self) -> dict[str, list[str] | str]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FlowSchema:
        return cls(
            numeric=list(data.get("numeric", NUMERIC_FEATURES)),
            categorical=list(data.get("categorical", CATEGORICAL_FEATURES)),
            label=data.get("label", LABEL_COLUMN),
        )
