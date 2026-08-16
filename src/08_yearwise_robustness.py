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

RESULTS_DIR = PROJECT_DIR / "results"


# =========================================================
# 2. Load prediction files
# =========================================================

persistence = pd.read_csv(
    RESULTS_DIR / "persistence_predictions.csv"
)

rf = pd.read_csv(
    RESULTS_DIR / "random_forest_predictions.csv"
)

xgb = pd.read_csv(
    RESULTS_DIR / "xgboost_predictions.csv"
)

lstm = pd.read_csv(
    RESULTS_DIR / "lstm_predictions.csv"
)


# =========================================================
# 3. Convert dates
# =========================================================

for df in [
    persistence,
    rf,
    xgb,
    lstm
]:
    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )


# =========================================================
# 4. Basic row validation
# =========================================================

print("\n========================================")
print("YEAR-WISE ROBUSTNESS ANALYSIS")
print("========================================")

print("\nRows:")

print(
    "Persistence   :",
    len(persistence)
)

print(
    "Random Forest :",
    len(rf)
)

print(
    "XGBoost       :",
    len(xgb)
)

print(
    "LSTM          :",
    len(lstm)
)


# =========================================================
# 5. Keep master actual values from persistence
# =========================================================

master = persistence[
    [
        "target_date",
        "point_id",
        "rainfall_next_day",
        "temperature_next_day",
        "rain_pred_persistence",
        "temp_pred_persistence"
    ]
].copy()


# =========================================================
# 6. Add Random Forest predictions
# =========================================================

master = master.merge(

    rf[
        [
            "target_date",
            "point_id",
            "rain_pred_rf",
            "temp_pred_rf"
        ]
    ],

    on=[
        "target_date",
        "point_id"
    ],

    how="inner",

    validate="one_to_one"
)


# =========================================================
# 7. Add XGBoost predictions
# =========================================================

master = master.merge(

    xgb[
        [
            "target_date",
            "point_id",
            "rain_pred_xgb",
            "temp_pred_xgb"
        ]
    ],

    on=[
        "target_date",
        "point_id"
    ],

    how="inner",

    validate="one_to_one"
)


# =========================================================
# 8. Add LSTM predictions
# =========================================================

master = master.merge(

    lstm[
        [
            "target_date",
            "point_id",
            "rain_pred_lstm",
            "temp_pred_lstm"
        ]
    ],

    on=[
        "target_date",
        "point_id"
    ],

    how="inner",

    validate="one_to_one"
)


# =========================================================
# 9. Validate merged predictions
# =========================================================

print("\nMerged prediction rows:", len(master))

print(
    "Missing values:",
    master.isnull().sum().sum()
)

print(
    "Duplicate date-point records:",
    master.duplicated(
        [
            "target_date",
            "point_id"
        ]
    ).sum()
)


# =========================================================
# 10. Define July-June climate year
# =========================================================

def assign_climate_year(date):

    if date.month >= 7:
        start_year = date.year

    else:
        start_year = date.year - 1

    end_year = start_year + 1

    return (
        f"{start_year}-"
        f"{str(end_year)[-2:]}"
    )


master["climate_year"] = (
    master["target_date"]
    .apply(assign_climate_year)
)


# =========================================================
# 11. Check observations per year
# =========================================================

print("\nRows per climate year:")

print(
    master[
        "climate_year"
    ].value_counts().sort_index()
)


# =========================================================
# 12. Metric function
# =========================================================

def calculate_metrics(
    actual,
    predicted
):

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
# 13. Prediction columns
# =========================================================

models = {

    "Persistence": {
        "rain":
            "rain_pred_persistence",
        "temp":
            "temp_pred_persistence"
    },

    "Random Forest": {
        "rain":
            "rain_pred_rf",
        "temp":
            "temp_pred_rf"
    },

    "XGBoost": {
        "rain":
            "rain_pred_xgb",
        "temp":
            "temp_pred_xgb"
    },

    "LSTM": {
        "rain":
            "rain_pred_lstm",
        "temp":
            "temp_pred_lstm"
    }

}


# =========================================================
# 14. Calculate year-wise performance
# =========================================================

results = []


for climate_year, year_data in (
    master.groupby("climate_year")
):

    print(
        "\n========================================"
    )

    print(
        "CLIMATE YEAR:",
        climate_year
    )

    print(
        "Rows:",
        len(year_data)
    )

    print(
        "========================================"
    )


    for model_name, columns in models.items():

        # -----------------------------------------------
        # Rainfall
        # -----------------------------------------------

        rain_mae, rain_rmse, rain_r2 = (
            calculate_metrics(

                year_data[
                    "rainfall_next_day"
                ],

                year_data[
                    columns["rain"]
                ]
            )
        )


        results.append({

            "climate_year":
                climate_year,

            "model":
                model_name,

            "target":
                "Rainfall",

            "MAE":
                rain_mae,

            "RMSE":
                rain_rmse,

            "R2":
                rain_r2

        })


        # -----------------------------------------------
        # Temperature
        # -----------------------------------------------

        temp_mae, temp_rmse, temp_r2 = (
            calculate_metrics(

                year_data[
                    "temperature_next_day"
                ],

                year_data[
                    columns["temp"]
                ]
            )
        )


        results.append({

            "climate_year":
                climate_year,

            "model":
                model_name,

            "target":
                "Temperature",

            "MAE":
                temp_mae,

            "RMSE":
                temp_rmse,

            "R2":
                temp_r2

        })


# =========================================================
# 15. Results dataframe
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# 16. Save results
# =========================================================

output_file = (
    RESULTS_DIR /
    "yearwise_metrics.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# 17. Display rainfall results
# =========================================================

print("\n========================================")
print("YEAR-WISE RAINFALL RESULTS")
print("========================================")

rain_results = results_df[
    results_df["target"] == "Rainfall"
]

print(

    rain_results.to_string(
        index=False
    )

)


# =========================================================
# 18. Display temperature results
# =========================================================

print("\n========================================")
print("YEAR-WISE TEMPERATURE RESULTS")
print("========================================")

temp_results = results_df[
    results_df["target"] == "Temperature"
]

print(

    temp_results.to_string(
        index=False
    )

)


print(
    "\nSaved:",
    output_file
)