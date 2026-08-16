import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"

files = [
    "persistence_predictions.csv",
    "random_forest_predictions.csv",
    "xgboost_predictions.csv",
    "lstm_predictions.csv"
]

for filename in files:

    path = RESULTS / filename

    if not path.exists():
        print(f"\nMissing: {filename}")
        continue

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    print("Columns:")
    print(df.columns.tolist())

    rainfall_prediction_columns = [
        c for c in df.columns
        if "rain" in c.lower()
        and "pred" in c.lower()
    ]

    if not rainfall_prediction_columns:
        print("No rainfall prediction column automatically detected.")
        continue

    for col in rainfall_prediction_columns:

        negative = (df[col] < 0).sum()

        print(f"\nColumn: {col}")
        print(f"Minimum prediction: {df[col].min():.6f}")
        print(f"Negative predictions: {negative}")
        print(
            f"Percentage negative: "
            f"{100 * negative / len(df):.4f}%"
        )