
# ==========================================================
# TRACE: MAIN PIPELINE
# Executes every stage in the correct order
# ==========================================================

import subprocess
import sys
from pathlib import Path


# SECTION 1: LOCATE THE PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# SECTION 2: DEFINE THE WORKFLOW
STEPS = [
    ("Generate telemetry", "src/data_generator.py"),
    ("Engineer features", "src/features.py"),
    ("Detect anomalies", "src/anomaly_detector.py"),
    ("Rank root causes", "src/root_cause.py"),
]


# SECTION 3: RUN EACH STAGE
def run_pipeline():
    print("\n========== TRACE STARTED ==========")

    for step_name, script_path in STEPS:
        print(f"\nRunning: {step_name}")

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"TRACE stopped because this stage failed: {step_name}"
            )

    print("\n========== TRACE COMPLETED ==========")
    print("Updated root-cause ranking:")
    print(PROJECT_ROOT / "reports" / "root_cause_ranking.csv")


# SECTION 4: START THE PIPELINE
if __name__ == "__main__":
    run_pipeline()
