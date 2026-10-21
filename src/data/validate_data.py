from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "clean_apartments.csv"


def validate_data():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())

    required_columns = [
        "street_type",
        "street_name",
        "neighborhood",
        "area_sqm",
        "price_lei",
        "year_built",
        "transaction_period",
        "price_per_sqm",
        "transaction_quarter"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("Column validation passed!")
    if (df["area_sqm"] < 10).any():
        raise ValueError("Invalid apartment area detected.")

    if (df["price_lei"] <= 0).any():
        raise ValueError("Invalid transaction price detected.")

    if not df["transaction_quarter"].isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid transaction quarter detected.")

    print("Numeric validation passed!")

    required_non_null = [
        "area_sqm",
        "price_lei",
        "neighborhood",
        "street_type",
        "transaction_quarter"
    ]

    missing_values = df[required_non_null].isna().sum()

    if (missing_values > 0).any():
        raise ValueError(
            f"Missing values detected:\n{missing_values}"
        )

    print("Missing values validation passed!")

    numeric_columns = [
        "area_sqm",
        "price_lei",
        "year_built",
        "price_per_sqm",
        "transaction_quarter"
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )

    print("Data types validation passed!")

if __name__ == "__main__":
    validate_data()
