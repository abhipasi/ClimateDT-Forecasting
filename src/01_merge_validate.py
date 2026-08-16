import pandas as pd


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

rain = pd.read_csv(
    "data/CHIRPS_Maharashtra_20160701_20260630.csv"
)

temp = pd.read_csv(
    "data/ERA5_Maharashtra_20160701_20260630.csv"
)

points = pd.read_csv(
    "data/Maharashtra_Climate_Sampling_Points.csv"
)


# ---------------------------------------------------------
# 2. Convert date
# ---------------------------------------------------------

rain["date"] = pd.to_datetime(rain["date"])
temp["date"] = pd.to_datetime(temp["date"])


# ---------------------------------------------------------
# 3. Basic information
# ---------------------------------------------------------

print("\n--- CHIRPS ---")
print("Rows:", len(rain))
print("Dates:", rain["date"].nunique())
print("Points:", rain["point_id"].nunique())
print("Missing:", rain.isnull().sum())

print("\n--- ERA5-Land ---")
print("Rows:", len(temp))
print("Dates:", temp["date"].nunique())
print("Points:", temp["point_id"].nunique())
print("Missing:", temp.isnull().sum())


# ---------------------------------------------------------
# 4. Remove source columns before merge
# ---------------------------------------------------------

rain = rain.drop(columns=["source"])
temp = temp.drop(columns=["source"])


# ---------------------------------------------------------
# 5. Merge using date + point_id
# ---------------------------------------------------------

climate = pd.merge(
    rain,
    temp[
        [
            "date",
            "point_id",
            "temperature_c"
        ]
    ],
    on=[
        "date",
        "point_id"
    ],
    how="inner"
)


# ---------------------------------------------------------
# 6. Sort
# ---------------------------------------------------------

climate = climate.sort_values(
    ["point_id", "date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 7. Validate merged dataset
# ---------------------------------------------------------

print("\n--- MERGED DATASET ---")

print("Rows:", len(climate))

print(
    "Unique dates:",
    climate["date"].nunique()
)

print(
    "Unique points:",
    climate["point_id"].nunique()
)

print(
    "Duplicate date-point records:",
    climate.duplicated(
        ["date", "point_id"]
    ).sum()
)

print("\nMissing values:")

print(
    climate.isnull().sum()
)


# ---------------------------------------------------------
# 8. Descriptive statistics
# ---------------------------------------------------------

print("\n--- CLIMATE STATISTICS ---")

print(
    climate[
        [
            "rainfall_mm",
            "temperature_c"
        ]
    ].describe()
)


# ---------------------------------------------------------
# 9. Save master dataset
# ---------------------------------------------------------

climate.to_csv(
    "data/climate_master.csv",
    index=False
)

print(
    "\nSaved: data/climate_master.csv"
)