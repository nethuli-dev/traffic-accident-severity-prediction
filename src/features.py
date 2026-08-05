"""
Reusable feature engineering functions for the traffic accident severity project.
All functions are deterministic (rule-based) — safe to apply independently to
train/val/test without any risk of leakage.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, time_col: str = "Start_Time") -> pd.DataFrame:
    """Add hour bucket, day of week, weekend flag, rush hour flag, season."""
    df = df.copy()

    df["hour"] = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # Rush hour: weekday 7-9am or 4-6pm
    is_weekday = ~df["is_weekend"]
    is_morning_rush = df["hour"].between(7, 9)
    is_evening_rush = df["hour"].between(16, 18)
    df["is_rush_hour"] = is_weekday & (is_morning_rush | is_evening_rush)

    def hour_bucket(h):
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 17:
            return
