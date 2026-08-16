import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_squared_error


# =========================================================
# 1. GLOBAL PUBLICATION SETTINGS
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["lines.markersize"] = 5

FIGURE_DPI = 600


# =========================================================
# 2. PROJECT FOLDERS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures" / "publication"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 3. HELPER FUNCTION FOR SAVING
# =========================================================

def save_figure(fig, filename):

    png_file = FIGURES_DIR / f"{filename}.png"
    pdf_file = FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(
        png_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", png_file)
    print("Saved:", pdf_file)


# =========================================================
# 4. LOAD OVERALL MODEL METRICS
# =========================================================

metric_files = [

    RESULTS_DIR / "persistence_metrics.csv",

    RESULTS_DIR / "random_forest_metrics.csv",

    RESULTS_DIR / "xgboost_metrics.csv",

    RESULTS_DIR / "lstm_metrics.csv"
]


metric_frames = []

for file in metric_files:

    if not file.exists():
        raise FileNotFoundError(
            f"Missing metrics file:\n{file}"
        )

    temp = pd.read_csv(file)

    metric_frames.append(temp)


metrics = pd.concat(
    metric_frames,
    ignore_index=True
)


print("\n=====================================")
print("OVERALL METRICS")
print("=====================================")

print(metrics.to_string(index=False))


# =========================================================
# 5. CONTROL MODEL ORDER
# =========================================================

model_order = [

    "Persistence",
    "Random Forest",
    "XGBoost",
    "LSTM"
]


metrics["model"] = pd.Categorical(

    metrics["model"],

    categories=model_order,

    ordered=True
)


metrics = metrics.sort_values(
    "model"
)


# =========================================================
# FIGURE 3A
# OVERALL RAINFALL RMSE
# =========================================================

rain = metrics[
    metrics["target"].str.lower() == "rainfall"
].copy()


fig, ax = plt.subplots(
    figsize=(5.5, 3.8)
)


bars = ax.bar(
    rain["model"].astype(str),
    rain["RMSE"],
    width=0.62
)


ax.set_ylabel(
    "RMSE (mm/day)"
)

ax.set_xlabel(
    "Forecasting model"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.set_axisbelow(True)


# Value labels
for bar, value in zip(
    bars,
    rain["RMSE"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()


save_figure(
    fig,
    "fig3a_rainfall_rmse"
)


# =========================================================
# FIGURE 3B
# OVERALL TEMPERATURE RMSE
# =========================================================

temp = metrics[
    metrics["target"].str.lower() == "temperature"
].copy()


fig, ax = plt.subplots(
    figsize=(5.5, 3.8)
)


bars = ax.bar(
    temp["model"].astype(str),
    temp["RMSE"],
    width=0.62
)


ax.set_ylabel(
    "RMSE (°C)"
)

ax.set_xlabel(
    "Forecasting model"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.set_axisbelow(True)


for bar, value in zip(
    bars,
    temp["RMSE"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()


save_figure(
    fig,
    "fig3b_temperature_rmse"
)


# =========================================================
# 6. LOAD ABLATION TEST RESULTS
# =========================================================

ablation_file = (
    RESULTS_DIR /
    "ablation_test.csv"
)

if not ablation_file.exists():
    raise FileNotFoundError(
        f"Ablation test file not found:\n{ablation_file}"
    )

ablation = pd.read_csv(
    ablation_file
)


print("\n=====================================")
print("ABLATION TEST DATA")
print("=====================================")

print("File:", ablation_file.name)
print("Columns:", ablation.columns.tolist())

print(
    ablation.to_string(
        index=False
    )
)


# =========================================================
# 6A. CLEAN FEATURE-SET LABELS FOR PUBLICATION
# =========================================================

label_map = {

    "A_Basic":
        "Basic",

    "B_Temporal_Lags":
        "+ Temporal lags",

    "C_Full_Engineered":
        "+ Full engineered"
}


ablation[
    "display_configuration"
] = ablation[
    "configuration"
].map(label_map)


# Check that every configuration was mapped
if ablation["display_configuration"].isna().any():

    missing_labels = (
        ablation.loc[
            ablation["display_configuration"].isna(),
            "configuration"
        ]
        .drop_duplicates()
        .tolist()
    )

    raise ValueError(
        "Unrecognized ablation configuration(s): "
        + ", ".join(missing_labels)
    )


# =========================================================
# FIGURE 4A
# RAINFALL ABLATION
# =========================================================

ab_rain = ablation[
    ablation["target"] == "Rainfall"
].copy()


fig, ax = plt.subplots(
    figsize=(5.5, 3.8)
)


ax.plot(
    ab_rain[
        "display_configuration"
    ],

    ab_rain[
        "RMSE"
    ],

    marker="o"
)


ax.set_xlabel(
    "Feature configuration"
)

ax.set_ylabel(
    "RMSE (mm/day)"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


for x, y in zip(

    ab_rain[
        "display_configuration"
    ],

    ab_rain[
        "RMSE"
    ]
):

    ax.annotate(
        f"{y:.3f}",

        xy=(x, y),

        xytext=(0, 7),

        textcoords="offset points",

        ha="center",

        fontsize=9
    )


fig.tight_layout()


save_figure(
    fig,
    "fig4a_rainfall_ablation"
)


# =========================================================
# FIGURE 4B
# TEMPERATURE ABLATION
# =========================================================

ab_temp = ablation[
    ablation["target"] == "Temperature"
].copy()


fig, ax = plt.subplots(
    figsize=(5.5, 3.8)
)


ax.plot(
    ab_temp[
        "display_configuration"
    ],

    ab_temp[
        "RMSE"
    ],

    marker="o"
)


ax.set_xlabel(
    "Feature configuration"
)

ax.set_ylabel(
    "RMSE (°C)"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


for x, y in zip(

    ab_temp[
        "display_configuration"
    ],

    ab_temp[
        "RMSE"
    ]
):

    ax.annotate(
        f"{y:.3f}",

        xy=(x, y),

        xytext=(0, 7),

        textcoords="offset points",

        ha="center",

        fontsize=9
    )


fig.tight_layout()


save_figure(
    fig,
    "fig4b_temperature_ablation"
)


# =========================================================
# 7. LOAD YEAR-WISE ROBUSTNESS RESULTS
# =========================================================

year_file = (
    RESULTS_DIR /
    "yearwise_metrics.csv"
)


yearwise = pd.read_csv(
    year_file
)


print("\n=====================================")
print("YEAR-WISE METRICS")
print("=====================================")

print(
    yearwise.to_string(
        index=False
    )
)


year_order = [
    "2023-24",
    "2024-25",
    "2025-26"
]


yearwise[
    "climate_year"
] = pd.Categorical(

    yearwise[
        "climate_year"
    ],

    categories=year_order,

    ordered=True
)


# =========================================================
# FIGURE 5A
# YEAR-WISE RAINFALL RMSE
# =========================================================

yrain = yearwise[
    yearwise["target"]
    .str.lower()
    .eq("rainfall")
].copy()


fig, ax = plt.subplots(
    figsize=(5.8, 3.9)
)


for model in model_order:

    subset = yrain[
        yrain["model"] == model
    ].sort_values(
        "climate_year"
    )


    ax.plot(
        subset[
            "climate_year"
        ].astype(str),

        subset[
            "RMSE"
        ],

        marker="o",

        label=model
    )


ax.set_xlabel(
    "Climate year"
)

ax.set_ylabel(
    "RMSE (mm/day)"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.legend(
    frameon=False,
    ncol=2
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()


save_figure(
    fig,
    "fig5a_yearwise_rainfall_rmse"
)


# =========================================================
# FIGURE 5B
# YEAR-WISE TEMPERATURE RMSE
# =========================================================

ytemp = yearwise[
    yearwise["target"]
    .str.lower()
    .eq("temperature")
].copy()


fig, ax = plt.subplots(
    figsize=(5.8, 3.9)
)


for model in model_order:

    subset = ytemp[
        ytemp["model"] == model
    ].sort_values(
        "climate_year"
    )


    ax.plot(
        subset[
            "climate_year"
        ].astype(str),

        subset[
            "RMSE"
        ],

        marker="o",

        label=model
    )


ax.set_xlabel(
    "Climate year"
)

ax.set_ylabel(
    "RMSE (°C)"
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)


ax.legend(
    frameon=False,
    ncol=2
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()


save_figure(
    fig,
    "fig5b_yearwise_temperature_rmse"
)


# =========================================================
# 8. LOAD ACTUAL VS PREDICTED DATA
# =========================================================

rf_pred = pd.read_csv(
    RESULTS_DIR /
    "random_forest_predictions.csv"
)

xgb_pred = pd.read_csv(
    RESULTS_DIR /
    "xgboost_predictions.csv"
)

lstm_pred = pd.read_csv(
    RESULTS_DIR /
    "lstm_predictions.csv"
)


for df in [
    rf_pred,
    xgb_pred,
    lstm_pred
]:

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )


# =========================================================
# 9. FIND MEDIAN-PERFORMING LSTM LOCATION
# =========================================================

point_rmse = []


for point_id, group in (
    lstm_pred.groupby(
        "point_id"
    )
):

    rmse = np.sqrt(

        mean_squared_error(

            group[
                "rainfall_next_day"
            ],

            group[
                "rain_pred_lstm"
            ]
        )
    )


    point_rmse.append(
        {
            "point_id": point_id,
            "rmse": rmse
        }
    )


point_rmse = pd.DataFrame(
    point_rmse
).sort_values(
    "rmse"
).reset_index(
    drop=True
)


median_index = (
    len(point_rmse) // 2
)


selected_point = (
    point_rmse
    .iloc[median_index]
    ["point_id"]
)


selected_rmse = (
    point_rmse
    .iloc[median_index]
    ["rmse"]
)


print("\n=====================================")
print("REPRESENTATIVE LOCATION SELECTION")
print("=====================================")

print(
    "Median-performing location:",
    selected_point
)

print(
    f"LSTM rainfall RMSE: "
    f"{selected_rmse:.4f} mm/day"
)


# Save ranking for reproducibility

point_rmse.to_csv(

    RESULTS_DIR /
    "lstm_pointwise_rainfall_rmse.csv",

    index=False
)


# =========================================================
# 10. FIXED MONSOON WINDOW
# =========================================================

START_DATE = pd.Timestamp(
    "2024-07-01"
)

END_DATE = pd.Timestamp(
    "2024-09-30"
)


# =========================================================
# 11. MERGE PREDICTIONS
# =========================================================

keys = [
    "target_date",
    "point_id"
]


comparison = rf_pred[
    [
        "target_date",
        "point_id",
        "rainfall_next_day",
        "temperature_next_day",
        "rain_pred_rf",
        "temp_pred_rf"
    ]
].copy()


comparison = comparison.merge(

    xgb_pred[
        [
            "target_date",
            "point_id",
            "rain_pred_xgb",
            "temp_pred_xgb"
        ]
    ],

    on=keys,

    how="inner"
)


comparison = comparison.merge(

    lstm_pred[
        [
            "target_date",
            "point_id",
            "rain_pred_lstm",
            "temp_pred_lstm"
        ]
    ],

    on=keys,

    how="inner"
)


window = comparison[

    (
        comparison[
            "point_id"
        ] == selected_point
    )

    &

    (
        comparison[
            "target_date"
        ] >= START_DATE
    )

    &

    (
        comparison[
            "target_date"
        ] <= END_DATE
    )

].copy()


window = window.sort_values(
    "target_date"
)


print(
    "\nRows in representative "
    "monsoon plot:",
    len(window)
)


# =========================================================
# FIGURE 6A
# ACTUAL VS PREDICTED RAINFALL
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 3.8)
)


ax.plot(
    window["target_date"],
    window["rainfall_next_day"],
    label="Observed",
    linewidth=1.5
)


ax.plot(
    window["target_date"],
    window["rain_pred_rf"],
    label="Random Forest",
    linewidth=1.0
)


ax.plot(
    window["target_date"],
    window["rain_pred_xgb"],
    label="XGBoost",
    linewidth=1.0
)


ax.plot(
    window["target_date"],
    window["rain_pred_lstm"],
    label="LSTM",
    linewidth=1.0
)


ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Rainfall (mm/day)"
)


ax.legend(
    frameon=False,
    ncol=2
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.35
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.autofmt_xdate(
    rotation=30,
    ha="right"
)


fig.tight_layout()


save_figure(
    fig,
    "fig6a_observed_predicted_rainfall"
)


# =========================================================
# FIGURE 6B
# ACTUAL VS PREDICTED TEMPERATURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 3.8)
)


ax.plot(
    window["target_date"],
    window["temperature_next_day"],
    label="Observed",
    linewidth=1.5
)


ax.plot(
    window["target_date"],
    window["temp_pred_rf"],
    label="Random Forest",
    linewidth=1.0
)


ax.plot(
    window["target_date"],
    window["temp_pred_xgb"],
    label="XGBoost",
    linewidth=1.0
)


ax.plot(
    window["target_date"],
    window["temp_pred_lstm"],
    label="LSTM",
    linewidth=1.0
)


ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Temperature (°C)"
)


ax.legend(
    frameon=False,
    ncol=2
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.4,
    alpha=0.35
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.autofmt_xdate(
    rotation=30,
    ha="right"
)


fig.tight_layout()


save_figure(
    fig,
    "fig6b_observed_predicted_temperature"
)


# =========================================================
# FINAL CONFIRMATION
# =========================================================

print("\n=====================================")
print("PUBLICATION FIGURES COMPLETE")
print("=====================================")

print(
    "Output folder:"
)

print(
    FIGURES_DIR
)

print(
    "\nPNG resolution: 600 DPI"
)

print(
    "Vector PDF copies also generated."
)

print(
    "Font: Times New Roman"
)

print(
    "\nRepresentative actual/predicted "
    "location:",
    selected_point
)

print(
    "Plot period:",
    START_DATE.date(),
    "to",
    END_DATE.date()
)