"""
feature_engineering.py — Feature extraction helpers for model input.

Adds time-windowed statistical features on top of raw sensor readings.
These engineered features improve anomaly detection accuracy by capturing
trends and local variance.
"""

import numpy as np
import pandas as pd
from config.settings import SENSOR_COLUMNS


def add_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Add rolling mean and rolling std for each sensor column.
    Rows with insufficient history are forward-filled.
    """
    df = df.copy()
    for col in SENSOR_COLUMNS:
        df[f"{col}_roll_mean"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = (
            df[col].rolling(window=window, min_periods=1).std().fillna(0)
        )
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first-order differences (Δ) for each sensor.
    Captures rate-of-change — useful for detecting rapid fault onset.
    """
    df = df.copy()
    for col in SENSOR_COLUMNS:
        df[f"{col}_delta"] = df[col].diff().fillna(0)
    return df


def get_feature_columns(base_columns: list[str] | None = None,
                        include_rolling: bool = True,
                        include_delta: bool = True) -> list[str]:
    """Return the full list of feature column names produced by this module."""
    cols = list(base_columns or SENSOR_COLUMNS)
    if include_rolling:
        cols += [f"{c}_roll_mean" for c in SENSOR_COLUMNS]
        cols += [f"{c}_roll_std"  for c in SENSOR_COLUMNS]
    if include_delta:
        cols += [f"{c}_delta"     for c in SENSOR_COLUMNS]
    return cols


def build_feature_matrix(df: pd.DataFrame,
                          include_rolling: bool = True,
                          include_delta: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns DataFrame with engineered features only (no 'state' / metadata cols).
    """
    if include_rolling:
        df = add_rolling_stats(df)
    if include_delta:
        df = add_delta_features(df)

    feature_cols = get_feature_columns(
        include_rolling=include_rolling,
        include_delta=include_delta,
    )
    return df[feature_cols]
