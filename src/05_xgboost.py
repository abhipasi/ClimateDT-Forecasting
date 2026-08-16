import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor

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
# 2. Load datasets
# =========================================================

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


print("\n--- DATASETS ---")
print("Training rows   :", len(train))
print("Validation rows :", len(validation))
print("Testing rows    :", len(test))


# =========================================================
# 3. Features
# =========================================================

features = [

    "latitude",
    "longitude",

    "rainfall_mm",
    "temperature_c",

    "rain_lag1",
    "rain_lag3",
    "rain_lag7",

    "temp_lag1",
    "temp_lag3",
    "temp_lag7",

    "rain_roll7",
    "temp_roll7",

    "sin_doy",
    "cos_doy"
]


X_train = train[features]
X_val = validation[features]
X_test = test[features]


y_train_rain = train["rainfall_next_day"]
y_val_rain = validation["rainfall_next_day"]
y_test_rain = test["rainfall_next_day"]


y_train_temp = train["temperature_next_day"]
y_val_temp = validation["temperature_next_day"]
y_test_temp = test["temperature_next_day"]


# =========================================================
# 4. Metrics
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
# 5. XGBoost parameters
# =========================================================

common_params = {

    "n_estimators": 500,

    "learning_rate": 0.05,

    "max_depth": 6,

    "min_child_weight": 3,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "objective": "reg:squarederror",

    "random_state": 42,

    "n_jobs": -1
}


# =========================================================
# 6. Rainfall model
# =========================================================

print("\nTraining XGBoost for rainfall...")


rain_model = XGBRegressor(
    **common_params
)


rain_model.fit(
    X_train,
    y_train_rain
)


# ---------------------------------------------------------
# Validation predictions
# ---------------------------------------------------------

rain_val_pred_raw = rain_model.predict(X_val)


# Count negative raw predictions
negative_val_count = np.sum(
    rain_val_pred_raw < 0
)


print(
    "\nNegative rainfall predictions "
    "before clipping - validation:",
    negative_val_count
)


# ---------------------------------------------------------
# PHYSICAL CONSTRAINT
# Rainfall cannot be negative
# ---------------------------------------------------------

rain_val_pred = np.clip(
    rain_val_pred_raw,
    0,
    None
)


# Validation metrics are calculated AFTER clipping

rain_val_mae, rain_val_rmse, rain_val_r2 = (
    calculate_metrics(
        y_val_rain,
        rain_val_pred
    )
)


print("\n--- RAINFALL VALIDATION ---")

print(f"MAE  : {rain_val_mae:.4f} mm")
print(f"RMSE : {rain_val_rmse:.4f} mm")
print(f"R2   : {rain_val_r2:.4f}")


# ---------------------------------------------------------
# Test predictions
# ---------------------------------------------------------

rain_test_pred_raw = rain_model.predict(X_test)


# Count negative raw predictions
negative_test_count = np.sum(
    rain_test_pred_raw < 0
)


negative_test_percent = (
    negative_test_count /
    len(rain_test_pred_raw)
) * 100


print(
    "\nNegative rainfall predictions "
    "before clipping - test:",
    negative_test_count
)

print(
    f"Percentage negative before clipping: "
    f"{negative_test_percent:.4f}%"
)


print(
    f"Minimum raw rainfall prediction: "
    f"{rain_test_pred_raw.min():.6f} mm"
)


# ---------------------------------------------------------
# PHYSICAL CONSTRAINT
# ---------------------------------------------------------

rain_test_pred = np.clip(
    rain_test_pred_raw,
    0,
    None
)


print(
    f"Minimum rainfall prediction "
    f"after clipping: "
    f"{rain_test_pred.min():.6f} mm"
)


# Test metrics are calculated AFTER clipping

rain_mae, rain_rmse, rain_r2 = (
    calculate_metrics(
        y_test_rain,
        rain_test_pred
    )
)


# =========================================================
# 7. Temperature model
# =========================================================

print("\nTraining XGBoost for temperature...")


temp_model = XGBRegressor(
    **common_params
)


temp_model.fit(
    X_train,
    y_train_temp
)


# Validation

temp_val_pred = temp_model.predict(
    X_val
)


temp_val_mae, temp_val_rmse, temp_val_r2 = (
    calculate_metrics(
        y_val_temp,
        temp_val_pred
    )
)


print("\n--- TEMPERATURE VALIDATION ---")

print(
    f"MAE  : {temp_val_mae:.4f} °C"
)

print(
    f"RMSE : {temp_val_rmse:.4f} °C"
)

print(
    f"R2   : {temp_val_r2:.4f}"
)


# Test

temp_test_pred = temp_model.predict(
    X_test
)


temp_mae, temp_rmse, temp_r2 = (
    calculate_metrics(
        y_test_temp,
        temp_test_pred
    )
)


# =========================================================
# 8. Final test results
# =========================================================

print("\n========================================")
print("XGBOOST TEST RESULTS")
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
# 9. Save metrics
# =========================================================

metrics = pd.DataFrame([

    {
        "target": "Rainfall",
        "model": "XGBoost",
        "MAE": rain_mae,
        "RMSE": rain_rmse,
        "R2": rain_r2
    },

    {
        "target": "Temperature",
        "model": "XGBoost",
        "MAE": temp_mae,
        "RMSE": temp_rmse,
        "R2": temp_r2
    }

])


metrics.to_csv(
    RESULTS_DIR / "xgboost_metrics.csv",
    index=False
)


# =========================================================
# 10. Save predictions
# =========================================================

predictions = test[
    [
        "target_date",
        "point_id",
        "latitude",
        "longitude",
        "rainfall_next_day",
        "temperature_next_day"
    ]
].copy()


# Save both raw and physically corrected rainfall predictions

predictions[
    "rain_pred_xgb_raw"
] = rain_test_pred_raw


predictions[
    "rain_pred_xgb"
] = rain_test_pred


predictions[
    "temp_pred_xgb"
] = temp_test_pred


predictions.to_csv(
    RESULTS_DIR / "xgboost_predictions.csv",
    index=False
)


# =========================================================
# 11. Feature importance
# =========================================================

rain_importance = pd.DataFrame({

    "feature": features,

    "importance": rain_model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


temp_importance = pd.DataFrame({

    "feature": features,

    "importance": temp_model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


rain_importance.to_csv(
    RESULTS_DIR /
    "xgb_rain_feature_importance.csv",
    index=False
)


temp_importance.to_csv(
    RESULTS_DIR /
    "xgb_temp_feature_importance.csv",
    index=False
)


print("\nTop rainfall features:")
print(
    rain_importance.head(10)
)


print("\nTop temperature features:")
print(
    temp_importance.head(10)
)


print(
    "\nXGBoost results saved successfully."
)

print(
    "\nNOTE: Rainfall metrics were calculated "
    "after clipping negative predictions to zero."
)