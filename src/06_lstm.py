import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. Reproducibility
# =========================================================

SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# =========================================================
# 2. Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# 3. Load original master dataset
# =========================================================

df = pd.read_csv(
    DATA_DIR / "climate_master.csv"
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["point_id", "date"]
).reset_index(drop=True)


print("\n--- ORIGINAL DATA ---")

print("Rows   :", len(df))
print("Points :", df["point_id"].nunique())
print(
    "Period :",
    df["date"].min(),
    "to",
    df["date"].max()
)


# =========================================================
# 4. Seasonal features
# =========================================================

df["day_of_year"] = df["date"].dt.dayofyear

df["sin_doy"] = np.sin(
    2 * np.pi *
    df["day_of_year"] / 365.25
)

df["cos_doy"] = np.cos(
    2 * np.pi *
    df["day_of_year"] / 365.25
)


# =========================================================
# 5. Features used at every timestep
# =========================================================

FEATURES = [
    "rainfall_mm",
    "temperature_c",
    "latitude",
    "longitude",
    "sin_doy",
    "cos_doy"
]

SEQUENCE_LENGTH = 7


# =========================================================
# 6. Create sequence dataset
# =========================================================

X_sequences = []

y_rain = []
y_temp = []

target_dates = []
point_ids = []

latitudes = []
longitudes = []


for point_id, group in df.groupby("point_id"):

    group = group.sort_values(
        "date"
    ).reset_index(drop=True)

    values = group[FEATURES].values

    rainfall = group["rainfall_mm"].values
    temperature = group["temperature_c"].values
    dates = group["date"].values

    latitude = group["latitude"].values
    longitude = group["longitude"].values

    # Example:
    # Previous seven days -> predict following day

    for i in range(
        SEQUENCE_LENGTH - 1,
        len(group) - 1
    ):

        start = i - SEQUENCE_LENGTH + 1
        end = i + 1

        X_sequences.append(
            values[start:end]
        )

        y_rain.append(
            rainfall[i + 1]
        )

        y_temp.append(
            temperature[i + 1]
        )

        target_dates.append(
            dates[i + 1]
        )

        point_ids.append(
            point_id
        )

        latitudes.append(
            latitude[i]
        )

        longitudes.append(
            longitude[i]
        )


X = np.array(
    X_sequences,
    dtype=np.float32
)

y_rain = np.array(
    y_rain,
    dtype=np.float32
)

y_temp = np.array(
    y_temp,
    dtype=np.float32
)

target_dates = pd.to_datetime(
    np.array(target_dates)
)


print("\n--- SEQUENCE DATA ---")

print("X shape:", X.shape)

print(
    "Rainfall targets:",
    y_rain.shape
)

print(
    "Temperature targets:",
    y_temp.shape
)

print(
    "Target period:",
    target_dates.min(),
    "to",
    target_dates.max()
)


# =========================================================
# 7. Temporal split
# =========================================================

train_mask = (
    (target_dates >= pd.Timestamp("2016-07-09")) &
    (target_dates < pd.Timestamp("2022-07-01"))
)

val_mask = (
    (target_dates >= pd.Timestamp("2022-07-01")) &
    (target_dates < pd.Timestamp("2023-07-01"))
)

test_mask = (
    target_dates >= pd.Timestamp("2023-07-01")
)


X_train = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]


y_train_rain = y_rain[train_mask]
y_val_rain = y_rain[val_mask]
y_test_rain = y_rain[test_mask]


y_train_temp = y_temp[train_mask]
y_val_temp = y_temp[val_mask]
y_test_temp = y_temp[test_mask]


print("\n--- TEMPORAL SPLIT ---")

print(
    "Training sequences   :",
    len(X_train)
)

print(
    "Validation sequences :",
    len(X_val)
)

print(
    "Testing sequences    :",
    len(X_test)
)


# =========================================================
# 8. Scale input features
#
# Important:
# scaler is fitted ONLY on training data.
# =========================================================

n_features = X.shape[2]


X_scaler = StandardScaler()


X_train_2d = X_train.reshape(
    -1,
    n_features
)


X_scaler.fit(
    X_train_2d
)


def scale_sequences(data):

    original_shape = data.shape

    data_2d = data.reshape(
        -1,
        n_features
    )

    scaled = X_scaler.transform(
        data_2d
    )

    return scaled.reshape(
        original_shape
    )


X_train_scaled = scale_sequences(
    X_train
)

X_val_scaled = scale_sequences(
    X_val
)

X_test_scaled = scale_sequences(
    X_test
)


# =========================================================
# 9. Scale target variables separately
# =========================================================

rain_scaler = StandardScaler()

temp_scaler = StandardScaler()


rain_scaler.fit(
    y_train_rain.reshape(-1, 1)
)

temp_scaler.fit(
    y_train_temp.reshape(-1, 1)
)


y_train_rain_scaled = rain_scaler.transform(
    y_train_rain.reshape(-1, 1)
).flatten()

y_val_rain_scaled = rain_scaler.transform(
    y_val_rain.reshape(-1, 1)
).flatten()


y_train_temp_scaled = temp_scaler.transform(
    y_train_temp.reshape(-1, 1)
).flatten()

y_val_temp_scaled = temp_scaler.transform(
    y_val_temp.reshape(-1, 1)
).flatten()


# =========================================================
# 10. LSTM model function
# =========================================================

def build_lstm():

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                n_features
            )
        ),

        LSTM(
            64
        ),

        Dropout(
            0.20
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1
        )
    ])


    model.compile(

        optimizer="adam",

        loss="mse"

    )

    return model


# =========================================================
# 11. Early stopping
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1

)


# =========================================================
# 12. Rainfall LSTM
# =========================================================

print("\n========================================")
print("TRAINING RAINFALL LSTM")
print("========================================")


rain_model = build_lstm()


rain_history = rain_model.fit(

    X_train_scaled,

    y_train_rain_scaled,

    validation_data=(
        X_val_scaled,
        y_val_rain_scaled
    ),

    epochs=40,

    batch_size=128,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# =========================================================
# 13. Rainfall predictions
# =========================================================

rain_test_scaled = rain_model.predict(

    X_test_scaled,

    verbose=0
)


rain_test_pred = rain_scaler.inverse_transform(

    rain_test_scaled

).flatten()


# Rainfall cannot physically be negative
rain_test_pred = np.maximum(
    rain_test_pred,
    0
)


# =========================================================
# 14. Temperature LSTM
# =========================================================

print("\n========================================")
print("TRAINING TEMPERATURE LSTM")
print("========================================")


temp_model = build_lstm()


temp_history = temp_model.fit(

    X_train_scaled,

    y_train_temp_scaled,

    validation_data=(
        X_val_scaled,
        y_val_temp_scaled
    ),

    epochs=40,

    batch_size=128,

    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ],

    verbose=1
)


# =========================================================
# 15. Temperature predictions
# =========================================================

temp_test_scaled = temp_model.predict(

    X_test_scaled,

    verbose=0
)


temp_test_pred = temp_scaler.inverse_transform(

    temp_test_scaled

).flatten()


# =========================================================
# 16. Metric function
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
# 17. Test metrics
# =========================================================

rain_mae, rain_rmse, rain_r2 = (
    calculate_metrics(
        y_test_rain,
        rain_test_pred
    )
)


temp_mae, temp_rmse, temp_r2 = (
    calculate_metrics(
        y_test_temp,
        temp_test_pred
    )
)


# =========================================================
# 18. Print results
# =========================================================

print("\n========================================")
print("LSTM TEST RESULTS")
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
# 19. Save metrics
# =========================================================

metrics = pd.DataFrame([

    {
        "target": "Rainfall",
        "model": "LSTM",
        "MAE": rain_mae,
        "RMSE": rain_rmse,
        "R2": rain_r2
    },

    {
        "target": "Temperature",
        "model": "LSTM",
        "MAE": temp_mae,
        "RMSE": temp_rmse,
        "R2": temp_r2
    }

])


metrics.to_csv(

    RESULTS_DIR /
    "lstm_metrics.csv",

    index=False

)


# =========================================================
# 20. Save predictions
# =========================================================

test_metadata = pd.DataFrame({

    "target_date":
        target_dates[test_mask],

    "point_id":
        np.array(point_ids)[test_mask],

    "latitude":
        np.array(latitudes)[test_mask],

    "longitude":
        np.array(longitudes)[test_mask],

    "rainfall_next_day":
        y_test_rain,

    "rain_pred_lstm":
        rain_test_pred,

    "temperature_next_day":
        y_test_temp,

    "temp_pred_lstm":
        temp_test_pred

})


test_metadata.to_csv(

    RESULTS_DIR /
    "lstm_predictions.csv",

    index=False

)


# =========================================================
# 21. Save training histories
# =========================================================

pd.DataFrame(
    rain_history.history
).to_csv(

    RESULTS_DIR /
    "lstm_rain_history.csv",

    index=False
)


pd.DataFrame(
    temp_history.history
).to_csv(

    RESULTS_DIR /
    "lstm_temp_history.csv",

    index=False
)


# =========================================================
# 22. Save models
# =========================================================

rain_model.save(

    RESULTS_DIR /
    "lstm_rain.keras"

)


temp_model.save(

    RESULTS_DIR /
    "lstm_temperature.keras"

)


print(
    "\nLSTM results saved successfully."
)