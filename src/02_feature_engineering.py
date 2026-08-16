import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# 1. Project folders
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

input_file = DATA_DIR / "climate_master.csv"


# =========================================================
# 2. Load master dataset
# =========================================================

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["point_id", "date"]
).reset_index(drop=True)


print("\n--- ORIGINAL DATASET ---")
print("Rows:", len(df))
print("Points:", df["point_id"].nunique())
print("Dates:", df["date"].nunique())


# =========================================================
# 3. Calendar features
# =========================================================

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day_of_year"] = df["date"].dt.dayofyear


# Cyclical representation of season
df["sin_doy"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["cos_doy"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# =========================================================
# 4. Lag features
# =========================================================

grouped = df.groupby("point_id")


# Rainfall lags
df["rain_lag1"] = grouped["rainfall_mm"].shift(1)
df["rain_lag3"] = grouped["rainfall_mm"].shift(3)
df["rain_lag7"] = grouped["rainfall_mm"].shift(7)


# Temperature lags
df["temp_lag1"] = grouped["temperature_c"].shift(1)
df["temp_lag3"] = grouped["temperature_c"].shift(3)
df["temp_lag7"] = grouped["temperature_c"].shift(7)


# =========================================================
# 5. Seven-day rolling features
# =========================================================

df["rain_roll7"] = (
    df.groupby("point_id")["rainfall_mm"]
    .transform(
        lambda x: x.rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)

df["temp_roll7"] = (
    df.groupby("point_id")["temperature_c"]
    .transform(
        lambda x: x.rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)


# =========================================================
# 6. Create next-day prediction targets
# =========================================================

df["rainfall_next_day"] = (
    df.groupby("point_id")["rainfall_mm"]
    .shift(-1)
)

df["temperature_next_day"] = (
    df.groupby("point_id")["temperature_c"]
    .shift(-1)
)


# =========================================================
# 7. Target date
# =========================================================

df["target_date"] = df["date"] + pd.Timedelta(days=1)


# =========================================================
# 8. Check missing values introduced by lagging
# =========================================================

print("\n--- MISSING VALUES BEFORE CLEANING ---")

feature_columns = [
    "rain_lag1",
    "rain_lag3",
    "rain_lag7",
    "temp_lag1",
    "temp_lag3",
    "temp_lag7",
    "rain_roll7",
    "temp_roll7",
    "rainfall_next_day",
    "temperature_next_day"
]

print(df[feature_columns].isnull().sum())


# =========================================================
# 9. Remove rows where lag/target data are unavailable
# =========================================================

df_clean = df.dropna(
    subset=feature_columns
).copy()


# =========================================================
# 10. Validate cleaned feature dataset
# =========================================================

print("\n--- FEATURE DATASET ---")

print("Rows:", len(df_clean))

print(
    "Unique points:",
    df_clean["point_id"].nunique()
)

print(
    "Input date range:",
    df_clean["date"].min(),
    "to",
    df_clean["date"].max()
)

print(
    "Target date range:",
    df_clean["target_date"].min(),
    "to",
    df_clean["target_date"].max()
)

print(
    "Missing values:",
    df_clean.isnull().sum().sum()
)


# =========================================================
# 11. Chronological train-validation-test split
# =========================================================

train = df_clean[
    df_clean["target_date"] < "2022-07-01"
].copy()


validation = df_clean[
    (df_clean["target_date"] >= "2022-07-01") &
    (df_clean["target_date"] < "2023-07-01")
].copy()


test = df_clean[
    df_clean["target_date"] >= "2023-07-01"
].copy()


# =========================================================
# 12. Print split information
# =========================================================

print("\n--- TEMPORAL SPLIT ---")

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Training target dates:",
    train["target_date"].min(),
    "to",
    train["target_date"].max()
)


print(
    "\nValidation rows:",
    len(validation)
)

print(
    "Validation target dates:",
    validation["target_date"].min(),
    "to",
    validation["target_date"].max()
)


print(
    "\nTesting rows:",
    len(test)
)

print(
    "Testing target dates:",
    test["target_date"].min(),
    "to",
    test["target_date"].max()
)


# =========================================================
# 13. Save datasets
# =========================================================

feature_file = DATA_DIR / "climate_features.csv"
train_file = DATA_DIR / "train.csv"
validation_file = DATA_DIR / "validation.csv"
test_file = DATA_DIR / "test.csv"


df_clean.to_csv(
    feature_file,
    index=False
)

train.to_csv(
    train_file,
    index=False
)

validation.to_csv(
    validation_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


print("\nFiles saved successfully:")

print(feature_file)
print(train_file)
print(validation_file)
print(test_file)