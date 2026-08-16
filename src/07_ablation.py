import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
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
# 2. Load frozen datasets
# =========================================================

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


print("\n========================================")
print("ABLATION STUDY")
print("========================================")

print("\nTraining rows   :", len(train))
print("Validation rows :", len(validation))
print("Testing rows    :", len(test))


# =========================================================
# 3. Predefined feature configurations
# =========================================================

feature_sets = {

    # -----------------------------------------------------
    # Configuration A
    # Basic multi-source observations + geography
    # -----------------------------------------------------

    "A_Basic": [

        "latitude",
        "longitude",

        "rainfall_mm",
        "temperature_c"

    ],


    # -----------------------------------------------------
    # Configuration B
    # Basic + lagged temporal information
    # -----------------------------------------------------

    "B_Temporal_Lags": [

        "latitude",
        "longitude",

        "rainfall_mm",
        "temperature_c",

        "rain_lag1",
        "rain_lag3",
        "rain_lag7",

        "temp_lag1",
        "temp_lag3",
        "temp_lag7"

    ],


    # -----------------------------------------------------
    # Configuration C
    # Full data-engineering pipeline
    # -----------------------------------------------------

    "C_Full_Engineered": [

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
}


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
# 5. Fixed Random Forest settings
# =========================================================

def create_model():

    return RandomForestRegressor(

        n_estimators=200,

        max_depth=18,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )


# =========================================================
# 6. Results containers
# =========================================================

validation_results = []
test_results = []


# =========================================================
# 7. Run every configuration
# =========================================================

for config_name, features in feature_sets.items():

    print("\n========================================")
    print("CONFIGURATION:", config_name)
    print("========================================")

    print("Number of features:", len(features))


    # -----------------------------------------------------
    # Prepare input data
    # -----------------------------------------------------

    X_train = train[features]

    X_val = validation[features]

    X_test = test[features]


    # -----------------------------------------------------
    # RAINFALL
    # -----------------------------------------------------

    print("\nTraining rainfall model...")


    rain_model = create_model()

    rain_model.fit(

        X_train,

        train["rainfall_next_day"]

    )


    # Validation
    rain_val_pred = rain_model.predict(
        X_val
    )

    rain_val_mae, rain_val_rmse, rain_val_r2 = (
        calculate_metrics(

            validation["rainfall_next_day"],

            rain_val_pred
        )
    )


    # Test
    rain_test_pred = rain_model.predict(
        X_test
    )

    rain_test_mae, rain_test_rmse, rain_test_r2 = (
        calculate_metrics(

            test["rainfall_next_day"],

            rain_test_pred
        )
    )


    print("\nRainfall test:")
    print(f"MAE  : {rain_test_mae:.4f}")
    print(f"RMSE : {rain_test_rmse:.4f}")
    print(f"R2   : {rain_test_r2:.4f}")


    # -----------------------------------------------------
    # TEMPERATURE
    # -----------------------------------------------------

    print("\nTraining temperature model...")


    temp_model = create_model()

    temp_model.fit(

        X_train,

        train["temperature_next_day"]

    )


    # Validation
    temp_val_pred = temp_model.predict(
        X_val
    )

    temp_val_mae, temp_val_rmse, temp_val_r2 = (
        calculate_metrics(

            validation["temperature_next_day"],

            temp_val_pred
        )
    )


    # Test
    temp_test_pred = temp_model.predict(
        X_test
    )

    temp_test_mae, temp_test_rmse, temp_test_r2 = (
        calculate_metrics(

            test["temperature_next_day"],

            temp_test_pred
        )
    )


    print("\nTemperature test:")
    print(f"MAE  : {temp_test_mae:.4f}")
    print(f"RMSE : {temp_test_rmse:.4f}")
    print(f"R2   : {temp_test_r2:.4f}")


    # =====================================================
    # Store validation results
    # =====================================================

    validation_results.extend([

        {

            "configuration":
                config_name,

            "target":
                "Rainfall",

            "feature_count":
                len(features),

            "MAE":
                rain_val_mae,

            "RMSE":
                rain_val_rmse,

            "R2":
                rain_val_r2
        },

        {

            "configuration":
                config_name,

            "target":
                "Temperature",

            "feature_count":
                len(features),

            "MAE":
                temp_val_mae,

            "RMSE":
                temp_val_rmse,

            "R2":
                temp_val_r2
        }

    ])


    # =====================================================
    # Store test results
    # =====================================================

    test_results.extend([

        {

            "configuration":
                config_name,

            "target":
                "Rainfall",

            "feature_count":
                len(features),

            "MAE":
                rain_test_mae,

            "RMSE":
                rain_test_rmse,

            "R2":
                rain_test_r2
        },

        {

            "configuration":
                config_name,

            "target":
                "Temperature",

            "feature_count":
                len(features),

            "MAE":
                temp_test_mae,

            "RMSE":
                temp_test_rmse,

            "R2":
                temp_test_r2
        }

    ])


# =========================================================
# 8. Convert results to dataframes
# =========================================================

validation_df = pd.DataFrame(
    validation_results
)

test_df = pd.DataFrame(
    test_results
)


# =========================================================
# 9. Save
# =========================================================

validation_df.to_csv(

    RESULTS_DIR /
    "ablation_validation.csv",

    index=False
)


test_df.to_csv(

    RESULTS_DIR /
    "ablation_test.csv",

    index=False
)


# =========================================================
# 10. Display final results
# =========================================================

print("\n========================================")
print("ABLATION TEST RESULTS")
print("========================================")

print(

    test_df.to_string(
        index=False
    )

)


print(
    "\nSaved:"
)

print(
    RESULTS_DIR /
    "ablation_validation.csv"
)

print(
    RESULTS_DIR /
    "ablation_test.csv"
)