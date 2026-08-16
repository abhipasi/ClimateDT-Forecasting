import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# GLOBAL FIGURE SETTINGS
# =========================================================

# Use Times New Roman for all figure text
plt.rcParams["font.family"] = "Times New Roman"

# Font sizes
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11

# Useful if mathematical symbols are used
plt.rcParams["mathtext.fontset"] = "stix"

# High-resolution output
FIGURE_DPI = 600


# =========================================================
# Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. Load overall model metrics
# =========================================================

persistence = pd.read_csv(
    RESULTS_DIR / "persistence_metrics.csv"
)

rf = pd.read_csv(
    RESULTS_DIR / "random_forest_metrics.csv"
)

xgb = pd.read_csv(
    RESULTS_DIR / "xgboost_metrics.csv"
)

lstm = pd.read_csv(
    RESULTS_DIR / "lstm_metrics.csv"
)


metrics = pd.concat(
    [persistence, rf, xgb, lstm],
    ignore_index=True
)


# =========================================================
# 2. Rainfall RMSE comparison
# =========================================================

rain = metrics[
    metrics["target"] == "Rainfall"
]

plt.figure(figsize=(7, 5))

plt.bar(
    rain["model"],
    rain["RMSE"]
)

plt.ylabel("RMSE (mm/day)")
plt.xlabel("Model")

plt.title(
    "One-Day-Ahead Rainfall Forecasting"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_rainfall_rmse.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 3. Temperature RMSE comparison
# =========================================================

temp = metrics[
    metrics["target"] == "Temperature"
]

plt.figure(figsize=(7, 5))

plt.bar(
    temp["model"],
    temp["RMSE"]
)

plt.ylabel("RMSE (°C)")
plt.xlabel("Model")

plt.title(
    "One-Day-Ahead Temperature Forecasting"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_temperature_rmse.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 4. Ablation study
# =========================================================

ablation = pd.read_csv(
    RESULTS_DIR / "ablation_test.csv"
)


rain_ablation = ablation[
    ablation["target"] == "Rainfall"
]

plt.figure(figsize=(7, 5))

plt.bar(
    rain_ablation["configuration"],
    rain_ablation["RMSE"]
)

plt.ylabel("Rainfall RMSE (mm/day)")
plt.xlabel("Feature Configuration")

plt.title(
    "Effect of Spatiotemporal Feature Engineering"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "ablation_rainfall.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 5. Year-wise rainfall robustness
# =========================================================

yearwise = pd.read_csv(
    RESULTS_DIR / "yearwise_metrics.csv"
)

rain_year = yearwise[
    yearwise["target"] == "Rainfall"
]


plt.figure(figsize=(8, 5))


for model in rain_year["model"].unique():

    subset = rain_year[
        rain_year["model"] == model
    ]

    plt.plot(
        subset["climate_year"],
        subset["RMSE"],
        marker="o",
        linewidth=1.8,
        label=model
    )


plt.ylabel("Rainfall RMSE (mm/day)")
plt.xlabel("Climate Year")

plt.title(
    "Year-Wise Rainfall Forecasting Robustness"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "yearwise_rainfall_rmse.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 6. Year-wise temperature robustness
# =========================================================

temp_year = yearwise[
    yearwise["target"] == "Temperature"
]


plt.figure(figsize=(8, 5))


for model in temp_year["model"].unique():

    subset = temp_year[
        temp_year["model"] == model
    ]

    plt.plot(
        subset["climate_year"],
        subset["RMSE"],
        marker="o",
        linewidth=1.8,
        label=model
    )


plt.ylabel("Temperature RMSE (°C)")
plt.xlabel("Climate Year")

plt.title(
    "Year-Wise Temperature Forecasting Robustness"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "yearwise_temperature_rmse.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 7. Actual versus predicted rainfall for LSTM
# =========================================================

lstm_pred = pd.read_csv(
    RESULTS_DIR / "lstm_predictions.csv"
)

lstm_pred["target_date"] = pd.to_datetime(
    lstm_pred["target_date"]
)


# Use one location to keep the plot readable
first_point = (
    lstm_pred["point_id"]
    .sort_values()
    .iloc[0]
)

example = lstm_pred[
    lstm_pred["point_id"] == first_point
].copy()


# Select first 180 test days
example = example.head(180)


plt.figure(figsize=(10, 5))

plt.plot(
    example["target_date"],
    example["rainfall_next_day"],
    linewidth=1.5,
    label="Observed"
)

plt.plot(
    example["target_date"],
    example["rain_pred_lstm"],
    linewidth=1.5,
    label="LSTM Prediction"
)

plt.xlabel("Date")
plt.ylabel("Rainfall (mm/day)")

plt.title(
    f"Observed and Predicted Rainfall – {first_point}"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "rainfall_actual_vs_predicted.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 8. Actual versus predicted temperature
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    example["target_date"],
    example["temperature_next_day"],
    linewidth=1.5,
    label="Observed"
)

plt.plot(
    example["target_date"],
    example["temp_pred_lstm"],
    linewidth=1.5,
    label="LSTM Prediction"
)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")

plt.title(
    f"Observed and Predicted Temperature – {first_point}"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "temperature_actual_vs_predicted.png",
    dpi=FIGURE_DPI,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Final confirmation
# =========================================================

print("\nFigures generated successfully at 600 DPI.")
print("Font: Times New Roman")

for file in FIGURES_DIR.glob("*.png"):
    print(file.name)