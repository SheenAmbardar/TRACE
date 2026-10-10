
# ==========================================================
# TRACE: MODEL VALIDATION
# Evaluates anomaly detection against the known injected event.
# ==========================================================

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

# Locate the TRACE project and detection results
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "data" / "detections.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_validation.txt"


def main():
    # Load the model's predictions
    df = pd.read_csv(INPUT_PATH)

    # Label only the deliberately injected payment incident
    actual = (
        (df["service"] == "payment")
        & df["minute"].between(100, 110)
    ).astype(int)

    # Read the Isolation Forest predictions
    predicted = df["ml_anomaly"].astype(int)

    # Calculate the confusion matrix
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    # Generate precision, recall, and F1-score
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Injected incident"],
        zero_division=0,
    )

    # Prepare a readable validation report
    output = (
        "TRACE MODEL VALIDATION\n"
        "======================\n\n"
        "Confusion matrix [normal, incident]:\n"
        f"{matrix}\n\n"
        "Classification report:\n"
        f"{report}\n\n"
        "Note: Only the deliberately injected incident is labeled. "
        "Other real-world anomalies may be unlabeled.\n"
    )

    # Save the report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")

    print(output)
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
