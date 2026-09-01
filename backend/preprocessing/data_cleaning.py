"""
data_cleaning.py — Data cleaning utilities for raw sensor data.

Tasks:
  1. Detect and handle missing values
  2. Detect and clip sensor outliers
  3. Detect sensor freeze (stuck values)
"""

import pandas as pd
import numpy as np
from config.settings import SENSOR_SCHEMA, SENSOR_COLUMNS


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop rows where more than `threshold` fraction of sensor columns are NaN.
    Remaining NaN cells are forward-filled then back-filled.
    """
    sensor_df = df[SENSOR_COLUMNS]
    nan_fraction = sensor_df.isna().mean(axis=1)
    df = df[nan_fraction <= threshold].copy()
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].ffill().bfill()
    return df


def clip_outliers(df: pd.DataFrame, sigma: float = 4.0) -> pd.DataFrame:
    """
    Clip sensor values that are more than `sigma` standard deviations from
    the per-column mean (computed only on normal-range data if 'state' exists).
    """
    df = df.copy()
    for col in SENSOR_COLUMNS:
        mean = df[col].mean()
        std  = df[col].std()
        lower = mean - sigma * std
        upper = mean + sigma * std
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def detect_sensor_freeze(df: pd.DataFrame, window: int = 10, tolerance: float = 1e-6) -> pd.Series:
    """
    Returns a boolean Series indicating rows where any sensor has been
    frozen (constant value) for the last `window` ticks.
    """
    freeze_mask = pd.Series(False, index=df.index)
    for col in SENSOR_COLUMNS:
        rolling_std = df[col].rolling(window=window, min_periods=window).std()
        freeze_mask |= (rolling_std < tolerance)
    return freeze_mask


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: drop missing → clip outliers."""
    df = drop_missing(df)
    df = clip_outliers(df)
    return df
