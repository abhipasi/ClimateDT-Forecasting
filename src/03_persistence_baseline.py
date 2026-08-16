import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. Load test dataset
# =========================================================

test_file = DATA_DIR / "test.csv"

test = pd.read_csv(test_file)

test["date"] = pd.to_datetime(test["date"])
test["target_date"] = pd.to_datetime(test["target_date"])


print("\n--- TEST DATASET ---")

print("Rows:", len(test))

print(
    "Target period:",
    test["target_date"].min(),
    "to",
    test["target_date"].max()
)

print(
    "Points:",
    test["point_id"].nunique()
)


# =========================================================
# 3. Persistence predictions
# =========================================================

# Tomorrow rainfall = today's rainfall
test["rain_pred_persistence"] = test["rainfall_mm"]

# Tomorrow temperature = today's temperature
test["temp_pred_persistence"] = test["temperature_c"]


# =========================================================
# 4. Metric function
# =========================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# =========================================================
# 5. Rainfall metrics
# =========================================================

rain_mae, rain_rmse, rain_r2 = calculate_metrics(

    test["rainfall_next_day"],

    test["rain_pred_persistence"]
)


# =========================================================
# 6. Temperature metrics
# =========================================================

temp_mae, temp_rmse, temp_r2 = calculate_metrics(

    test["temperature_next_day"],

    test["temp_pred_persistence"]
)


# =========================================================
# 7. Display results
# =========================================================

print("\n========================================")
print("PERSISTENCE BASELINE RESULTS")
print("========================================")


print("\n--- RAINFALL ---")

print(
    f"MAE  : {rain_mae:.4f} mm"
)

print(
    f"RMSE : {rain_rmse:.4f} mm"
)

print(
    f"R2   : {rain_r2:.4f}"
)


print("\n--- TEMPERATURE ---")

print(
    f"MAE  : {temp_mae:.4f} °C"
)

print(
    f"RMSE : {temp_rmse:.4f} °C"
)

print(
    f"R2   : {temp_r2:.4f}"
)


# =========================================================
# 8. Save metrics
# =========================================================

metrics = pd.DataFrame([

    {
        "target": "Rainfall",
        "model": "Persistence",
        "MAE": rain_mae,
        "RMSE": rain_rmse,
        "R2": rain_r2
    },

    {
        "target": "Temperature",
        "model": "Persistence",
        "MAE": temp_mae,
        "RMSE": temp_rmse,
        "R2": temp_r2
    }

])


metrics_file = (
    RESULTS_DIR /
    "persistence_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# =========================================================
# 9. Save predictions
# =========================================================

prediction_columns = [

    "target_date",
    "point_id",
    "latitude",
    "longitude",

    "rainfall_next_day",
    "rain_pred_persistence",

    "temperature_next_day",
    "temp_pred_persistence"
]


predictions_file = (
    RESULTS_DIR /
    "persistence_predictions.csv"
)

test[
    prediction_columns
].to_csv(
    predictions_file,
    index=False
)


# =========================================================
# 10. Final confirmation
# =========================================================

print(
    "\nMetrics saved to:"
)

print(
    metrics_file
)

print(
    "\nPredictions saved to:"
)

print(
    predictions_file
)