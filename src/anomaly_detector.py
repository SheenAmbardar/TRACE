
# ==========================================================
# TRACE: ANOMALY DETECTION FOR DISTRIBUTED SYSTEMS
# Uses Isolation Forest to identify unusual service telemetry
# and evaluates predictions against a known synthetic incident.
# ==========================================================


# SECTION 1: IMPORT REQUIRED LIBRARIES
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


# ==========================================================
# SECTION 2: DEFINE INPUT AND OUTPUT PATHS
# ==========================================================

# Find the main TRACE project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read engineered features and save anomaly detection results.
INPUT_PATH = PROJECT_ROOT / "data" / "features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "detections.csv"


# ==========================================================
# SECTION 3: SELECT FEATURES FOR MACHINE LEARNING
# ==========================================================

# These telemetry measurements and z-scores are model inputs.
FEATURE_COLUMNS = [
    "latency_ms",
    "error_rate",
    "cpu_usage_pct",
    "latency_ms_zscore",
    "error_rate_zscore",
    "cpu_usage_pct_zscore",
]


# ==========================================================
# SECTION 4: LOAD AND VALIDATE THE ENGINEERED DATASET
# ==========================================================

def load_features():
    """Load engineered features and check required columns."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_PATH}. "
            "Run src/features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = set(FEATURE_COLUMNS) | {
        "service",
        "minute",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in features.csv: {missing}"
        )

    return df


# ==========================================================
# SECTION 5: BUILD THE ISOLATION FOREST PIPELINE
# ==========================================================

def create_detector():
    """Create a pipeline to handle missing values and detect anomalies."""

    detector = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            IsolationForest(
                n_estimators=200,
                contamination=0.03,
                random_state=42,
            ),
        ),
    ])

    return detector


# ==========================================================
# SECTION 6: FIT THE MODEL AND DETECT ANOMALIES
# ==========================================================

def detect_anomalies(df):
    """Fit Isolation Forest and append predictions and scores."""

    result = df.copy()

    # Select only the telemetry features used by the model.
    X = result[FEATURE_COLUMNS]

    # Create and fit the detector.
    detector = create_detector()

    # Isolation Forest predicts:
    #  1  = normal observation
    # -1  = anomalous observation
    predictions = detector.fit_predict(X)

    # Convert predictions into easier-to-read binary labels:
    # 1 = anomaly, 0 = normal.
    result["ml_anomaly"] = (predictions == -1).astype(int)

    # Higher scores indicate more unusual observations here.
    result["anomaly_score"] = -detector.decision_function(X)

    return result


# ==========================================================
# SECTION 7: EVALUATE AGAINST THE KNOWN SYNTHETIC INCIDENT
# ==========================================================

def evaluate_detections(result):
    """
    Evaluate predictions against the known injected incident.

    The incident labels are used only for evaluation.
    They are not provided to Isolation Forest during fitting.
    """

    # Ground truth:
    # payment service, minutes 100 through 110 inclusive.
    actual_incident = (
        (result["service"] == "payment")
        & result["minute"].between(100, 110)
    ).astype(int)

    # Predictions produced by the unsupervised ML model.
    predicted_incident = result["ml_anomaly"].astype(int)

    # Calculate classification metrics.
    precision = precision_score(
        actual_incident,
        predicted_incident,
        zero_division=0,
    )

    recall = recall_score(
        actual_incident,
        predicted_incident,
        zero_division=0,
    )

    f1 = f1_score(
        actual_incident,
        predicted_incident,
        zero_division=0,
    )

    # Calculate confusion-matrix counts.
    tn, fp, fn, tp = confusion_matrix(
        actual_incident,
        predicted_incident,
        labels=[0, 1],
    ).ravel()

    # Display the evaluation report.
    print("\n" + "=" * 55)
    print("TRACE SYNTHETIC INCIDENT EVALUATION")
    print("=" * 55)

    print("Known incident: payment, minutes 100–110")
    print(f"Actual incident records: {int(actual_incident.sum())}")

    print("\nConfusion matrix counts:")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives:  {tn}")
    print(f"False negatives: {fn}")

    print("\nEvaluation metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    print("=" * 55)

    # Return metrics for future dashboard or report integration.
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "actual_incident_records": int(actual_incident.sum()),
    }


# ==========================================================
# SECTION 8: SAVE AND SUMMARIZE THE DETECTIONS
# ==========================================================

def run_detection():
    """Run detection, evaluate it, and save the results."""

    # Step 1: Load the engineered telemetry dataset.
    df = load_features()

    # Step 2: Fit Isolation Forest and detect unusual records.
    result = detect_anomalies(df)

    # Step 3: Evaluate predictions against synthetic ground truth.
    evaluate_detections(result)

    # Step 4: Ensure the output directory exists.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 5: Save telemetry, predictions, and anomaly scores.
    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Step 6: Print a summary.
    print("\nTRACE Isolation Forest detection completed!")
    print(f"Rows analyzed: {len(result)}")
    print(f"ML anomalies detected: {int(result['ml_anomaly'].sum())}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nML anomaly counts by service:")

    print(
        result.groupby("service")["ml_anomaly"]
        .sum()
        .sort_values(ascending=False)
    )


# ==========================================================
# SECTION 9: EXECUTE THE MODULE
# ==========================================================

if __name__ == "__main__":
    run_detection()
