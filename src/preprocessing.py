"""Preprocessing utilities for Iris model training and prediction."""

from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


def build_preprocessing_pipeline() -> Pipeline:
    """Create a preprocessing pipeline for numeric Iris features."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )


def encode_target(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """Encode target species labels into numeric labels."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return pd.Series(encoded, index=y.index), encoder


def validate_measurements(values: dict) -> Tuple[bool, str]:
    """Validate input measurement dictionary and return validation status with message."""
    for field, value in values.items():
        if value is None:
            return False, f"{field.replace('_', ' ').title()} is required."
        if not isinstance(value, (int, float)):
            return False, f"{field.replace('_', ' ').title()} must be a number."
        if value <= 0:
            return False, f"{field.replace('_', ' ').title()} must be greater than zero."
        if value > 10:
            return False, f"{field.replace('_', ' ').title()} is outside the typical Iris range."
    return True, "OK"
