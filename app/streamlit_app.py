import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Real Estate Intelligence",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_xgb_model.joblib"
    return joblib.load(model_path)

model = load_model()
@st.cache_resource
def load_conformal_model():
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "conformal_model.joblib"
    )

    return joblib.load(model_path)


conformal_data = load_conformal_model()

conformal_model = conformal_data["model"]
conformal_margin = conformal_data["margin"]

print("Conformal margin:", conformal_margin)

@st.cache_resource
def load_similarity_model():
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "similarity_model.joblib"
    )

    return joblib.load(model_path)


similarity_data = load_similarity_model()

nn_model = similarity_data["model"]
similarity_preprocessor = similarity_data["preprocessor"]
X_reference = similarity_data["features"]
y_reference = similarity_data["prices"]

st.title("Real Estate Intelligence")

st.write(
    "AI-powered apartment price estimation "
    "using real estate transactions from "
    "Bucharest Sector 2 (2022)."
)

st.info(
    "Predictions are based on historical "
    "transactions from 2022, not current market prices."
)
st.subheader("Apartment Details")

with st.form("prediction_form"):
    area_sqm = st.number_input(
        "Apartment area (sqm)",
        min_value=10.0,
        max_value=500.0,
        value=60.0,
        step=1.0
    )

    year_built = st.number_input(
        "Construction year",
        min_value=1800,
        max_value=2022,
        value=1980
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "colentina",
            "pantelimon",
            "tei",
            "floreasca",
            "iancului",
            "centrala",
            "pipera sud",
            "vatra luminoasa",
            "andronache",
            "ion creanga",
            "unknown"
        ]
    )

    street_type = st.selectbox(
        "Street type",
        [
            "Strada",
            "Bulevard",
            "Alee",
            "Calea",
            "Sosea",
            "Intrare",
            "Piata",
            "Prelungire"
        ]
    )

    transaction_quarter = st.selectbox(
        "Transaction quarter",
        [1, 2, 3, 4]
    )

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    apartment = pd.DataFrame([{
        "street_type": street_type,
        "neighborhood": neighborhood,
        "area_sqm": area_sqm,
        "year_built": year_built,
        "transaction_quarter": transaction_quarter
    }])

    predicted_price = model.predict(apartment)[0]

    st.subheader("Estimated Apartment Price")

    st.metric(
        label="Predicted transaction price",
        value=f"{predicted_price:,.0f} RON"
    )
    conformal_prediction = conformal_model.predict(apartment)[0]

    lower_bound = conformal_prediction - conformal_margin
    upper_bound = conformal_prediction + conformal_margin

    st.subheader("90% Prediction Interval")

    st.write(
        f"Estimated range: {lower_bound:,.0f} - {upper_bound:,.0f} RON"
    )

    st.caption(
        "The interval is calculated using a separately calibrated "
        "XGBoost model trained on historical 2022 transactions. "
        "The observed test coverage was 88.01%, "
        "against a target of 90%."
    )
    query_transformed = similarity_preprocessor.transform(apartment)

    distances, indices = nn_model.kneighbors(
        query_transformed,
        n_neighbors=5
    )

    similar_apartments = X_reference.iloc[indices[0]].copy()

    similar_apartments["price_lei"] = y_reference.iloc[indices[0]].values
    similar_apartments["distance"] = distances[0]

    st.subheader("Similar Apartments")

    st.dataframe(
        similar_apartments[
            [
                "neighborhood",
                "area_sqm",
                "year_built",
                "price_lei",
                "distance"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )