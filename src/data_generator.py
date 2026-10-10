
from pathlib import Path
import numpy as np
import pandas as pd


# ==========================================================
# SECTION 1: CONFIGURE RANDOM DATA GENERATION
# ==========================================================

rng = np.random.default_rng(42)


# ==========================================================
# SECTION 2: DEFINE MICROSERVICES AND TIME DURATION
# ==========================================================

services = ["api", "order", "payment", "inventory"]

minutes = 180

records = []


# ==========================================================
# SECTION 3: GENERATE NORMAL TELEMETRY FOR EACH SERVICE
# ==========================================================

for service in services:
    for minute in range(minutes):

        latency = rng.normal(180, 15)
        error_rate = max(0, rng.normal(0.02, 0.01))
        cpu_usage = np.clip(rng.normal(45, 8), 0, 100)

        # Store one row of telemetry
        records.append({
            "minute": minute,
            "service": service,
            "latency_ms": round(latency, 2),
            "error_rate": round(error_rate, 4),
            "cpu_usage_pct": round(cpu_usage, 2),
        })


# ==========================================================
# SECTION 4: CONVERT RECORDS INTO A PANDAS DATAFRAME
# ==========================================================

df = pd.DataFrame(records)


# ==========================================================
# SECTION 5: INJECT A CONTROLLED PAYMENT-SERVICE INCIDENT
# ==========================================================

incident_mask = (
    (df["service"] == "payment")
    & (df["minute"].between(100, 110))
)

df.loc[incident_mask, "latency_ms"] *= 4
df.loc[incident_mask, "error_rate"] += 0.25
df.loc[incident_mask, "cpu_usage_pct"] = np.clip(
    df.loc[incident_mask, "cpu_usage_pct"] + 35,
    0,
    100,
)


# ==========================================================
# SECTION 6: LABEL INCIDENT ROWS FOR EVALUATION
# ==========================================================

df["is_incident"] = incident_mask.astype(int)


# ==========================================================
# SECTION 7: SAVE THE GENERATED DATASET
# ==========================================================

output_path = Path(__file__).resolve().parent.parent / "data" / "telemetry.csv"

df.to_csv(output_path, index=False)

print("TRACE telemetry dataset generated successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Services: {df['service'].nunique()}")
print(f"Incident rows: {df['is_incident'].sum()}")
print(f"Saved to: {output_path}")
