
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# SECTION 1: DEFINE FILE PATHS AND METRICS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "telemetry.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features.csv"

METRICS = ["latency_ms", "error_rate", "cpu_usage_pct"]


# ==========================================================
# SECTION 2: LOAD THE TELEMETRY DATASET
# ==========================================================

def load_telemetry():
    """Load telemetry and verify required columns."""

    df = pd.read_csv(INPUT_PATH)

    required_columns = ["minute", "service"] + METRICS
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values(
        ["service", "minute"]
    ).reset_index(drop=True)


# ==========================================================
# SECTION 3: CALCULATE HISTORICAL BASELINES
# ==========================================================

def add_rolling_features(df, window=15):
    """Calculate rolling statistics using previous observations only."""

    result = df.copy()

    for metric in METRICS:
        grouped = result.groupby("service")[metric]

        result[f"{metric}_rolling_mean"] = grouped.transform(
            lambda s: s.shift(1).rolling(
                window=window,
                min_periods=5
            ).mean()
        )

        result[f"{metric}_rolling_std"] = grouped.transform(
            lambda s: s.shift(1).rolling(
                window=window,
                min_periods=5
            ).std()
        )

        rolling_std = result[
            f"{metric}_rolling_std"
        ].replace(0, np.nan)

        result[f"{metric}_zscore"] = (
            result[metric] - result[f"{metric}_rolling_mean"]
        ) / rolling_std

    return result


# ==========================================================
# SECTION 4: FLAG STATISTICALLY UNUSUAL OBSERVATIONS
# ==========================================================

def add_anomaly_flags(df, threshold=3.0):
    """Flag measurements with absolute z-scores above the threshold."""

    result = df.copy()
    anomaly_columns = []

    for metric in METRICS:
        column = f"{metric}_anomaly"

        result[column] = (
            result[f"{metric}_zscore"].abs() > threshold
        ).fillna(False).astype(int)

        anomaly_columns.append(column)

    result["anomaly_count"] = result[anomaly_columns].sum(axis=1)

    return result


# ==========================================================
# SECTION 5: RUN FEATURE ENGINEERING AND SAVE THE CSV
# ==========================================================

def build_features():
    df = load_telemetry()
    df = add_rolling_features(df)
    df = add_anomaly_flags(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("TRACE feature engineering completed!")
    print(f"Feature dataset shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nAnomaly counts by service:")

    print(
        df.groupby("service")["anomaly_count"]
        .sum()
        .sort_values(ascending=False)
    )


# ==========================================================
# SECTION 6: EXECUTE WHEN THIS FILE IS RUN DIRECTLY
# ==========================================================

if __name__ == "__main__":
    build_features()
