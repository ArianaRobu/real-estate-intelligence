from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "export-tranzactii-2022-sector-2.csv"
)

PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "clean_apartments.csv"
)
def prepare_data():
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    print("Original dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    apartments = df[df["tip"] == "apartament"].copy()

    print("Apartments shape:", apartments.shape)

    apartments = apartments.rename(columns={
        "tip": "property_type",
        "Tip strada": "street_type",
        "Denumire strada": "street_name",
        "Zona": "neighborhood",
        "S utila": "area_sqm",
        "S desfasurata": "built_area_sqm",
        "S teren": "land_area_sqm",
        "Pret tranzactie LEI": "price_lei",
        "an_constructie": "year_built",
        "Perioada tranzactie": "transaction_period"
    })

    print("Renamed columns:", apartments.columns.tolist())

    apartments["neighborhood"] = (
        apartments["neighborhood"]
        .str.strip()
        .str.lower()
        .replace("-", pd.NA)
    )

    apartments["year_built"] = apartments["year_built"].replace(
        2087, pd.NA
    )

    print("Missing values:")
    print(apartments.isna().sum())

    apartments = apartments.dropna(subset=["area_sqm"])

    print("After removing missing area:", apartments.shape)

    apartments = apartments.dropna(subset=["price_lei"])

    print("After removing missing price:", apartments.shape)

    apartments["price_per_sqm"] = (
        apartments["price_lei"] / apartments["area_sqm"]
    )

    apartments = apartments[apartments["area_sqm"] >= 10].copy()

    print("After area filtering:", apartments.shape)

    apartments = apartments[
        apartments["price_per_sqm"] >= 2000
    ].copy()

    print("After price per sqm filtering:", apartments.shape)

    apartments = apartments.drop(columns=[
        "property_type",
        "built_area_sqm",
        "land_area_sqm"
    ])

    print("After dropping columns:", apartments.shape)
    print("Remaining columns:", apartments.columns.tolist())

    apartments["transaction_quarter"] = (
        apartments["transaction_period"]
        .str.extract(r"T([1-4])")[0]
        .astype(int)
    )

    apartments["neighborhood"] = apartments["neighborhood"].fillna("unknown")

    print("Final shape:", apartments.shape)
    print("Missing neighborhoods:", apartments["neighborhood"].isna().sum())
    print("Quarters:", sorted(apartments["transaction_quarter"].unique()))

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    apartments.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("Processed dataset saved successfully!")
    print("Saved to:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    prepare_data()
   