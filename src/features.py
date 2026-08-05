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
            return "afternoon"
        elif 17 <= h < 21:
            return "evening"
        else:
            return "night"

    df["hour_bucket"] = df["hour"].apply(hour_bucket).astype("category")

    def month_to_season(m):
        if m in (12, 1, 2):
            return "winter"
        elif m in (3, 4, 5):
            return "spring"
        elif m in (6, 7, 8):
            return "summer"
        else:
            return "fall"

    df["season"] = df[time_col].dt.month.apply(month_to_season).astype("category")

    return df


# ---------------------------------------------------------------------------
# Weather bucketing
# ---------------------------------------------------------------------------

def bucket_weather(condition: pd.Series) -> pd.Series:
    """
    Collapse the raw Weather_Condition strings (~40+ distinct values) into a
    small set of meaningful groups using keyword matching. This is more robust
    than a hardcoded lookup table since it catches variants you haven't seen
    yet (e.g. 'Light Rain', 'Heavy Rain', 'Rain Showers' all -> 'rain').
    """
    s = condition.astype("string").str.lower().fillna("missing")

    conditions = [
        s.eq("missing"),
        s.str.contains("thunder|storm", na=False),
        s.str.contains("snow|sleet|ice|wintry", na=False),
        s.str.contains("rain|drizzle", na=False),
        s.str.contains("fog|mist|haze", na=False),
        s.str.contains("smoke|dust|sand|ash", na=False),
        s.str.contains("wind|squall|tornado", na=False),
        s.str.contains("clear|fair", na=False),
        s.str.contains("cloud|overcast", na=False),
    ]
    choices = [
        "missing", "storm", "snow", "rain", "fog",
        "smoke_dust", "wind", "clear", "cloudy",
    ]

    bucketed = np.select(conditions, choices, default="other")
    return pd.Series(bucketed, index=condition.index, dtype="category")


def add_weather_bucket(df: pd.DataFrame, col: str = "Weather_Condition") -> pd.DataFrame:
    df = df.copy()
    df["weather_bucket"] = bucket_weather(df[col])
    return df


# ---------------------------------------------------------------------------
# Road context
# ---------------------------------------------------------------------------

ROAD_FEATURE_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
]

def add_road_feature_count(df: pd.DataFrame, cols=ROAD_FEATURE_COLS) -> pd.DataFrame:
    """Sum of boolean road-context flags present at the location."""
    df = df.copy()
    present_cols = [c for c in cols if c in df.columns]
    df["num_road_features_present"] = df[present_cols].fillna(False).astype(int).sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Geographic region
# ---------------------------------------------------------------------------

US_REGION_MAP = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South",
    "VA": "South", "DC": "South", "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West",
    "UT": "West", "WY": "West", "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}

def add_region(df: pd.DataFrame, state_col: str = "State") -> pd.DataFrame:
    df = df.copy()
    df["region"] = df[state_col].astype("string").map(US_REGION_MAP).fillna("Other").astype("category")
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Visibility x Precipitation: low visibility during active precipitation
    # is hypothesized to be worse than either factor alone.
    df["visibility_x_precip"] = df["Visibility(mi)"].fillna(0) * df["Precipitation(in)"].fillna(0)

    # Rush hour x Junction: congestion at junctions during rush hour is
    # hypothesized to raise collision likelihood (though not necessarily severity).
    if "is_rush_hour" not in df.columns:
        raise KeyError("Run add_time_features() before add_interaction_features().")
    df["rush_hour_x_junction"] = df["is_rush_hour"].astype(int) * df["Junction"].fillna(False).astype(int)

    return df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in order."""
    df = add_time_features(df)
    df = add_weather_bucket(df)
    df = add_road_feature_count(df)
    df = add_region(df)
    df = add_interaction_features(df)
    return df
