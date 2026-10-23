# Real Estate Intelligence

**End-to-End Machine Learning for Apartment Price Prediction in Bucharest**

Real Estate Intelligence is a data science project that analyzes real residential property transactions and predicts apartment prices using machine learning.

The project uses publicly available transaction data from Bucharest's Sector 2 (2022) and includes data preprocessing, exploratory data analysis, feature engineering, model training and evaluation, explainable AI, uncertainty estimation, and an interactive Streamlit application.

## Project Overview

The project was developed to demonstrate an end-to-end machine learning workflow, from raw data preparation to an interactive prediction application.

**Key components:**
- Data cleaning and validation
- Exploratory data analysis (EDA)
- Feature engineering and preprocessing
- Machine learning model comparison and hyperparameter tuning
- Model explainability using SHAP
- Prediction intervals using split conformal prediction
- Similar apartment retrieval using nearest neighbors
- Interactive web application built with Streamlit

The final modeling dataset contains **3,584 apartment transactions**, with property characteristics such as usable area, construction year, neighborhood, street type, and transaction quarter.

## Main Features

### 1. Apartment Price Prediction
Predicts apartment transaction prices in Romanian lei (RON) using an optimized XGBoost regression model.

### 2. Exploratory Data Analysis
Analyzes apartment prices, usable area, construction years, neighborhoods, and transaction patterns.

### 3. Explainable AI
Uses SHAP to investigate which property characteristics contribute most to model predictions.

### 4. Prediction Uncertainty
Provides prediction intervals using split conformal prediction, with a target coverage of 90%.

### 5. Similar Apartment Search
Uses a nearest-neighbors algorithm to identify five comparable apartments from the training dataset.

### 6. Interactive Streamlit Application
Allows users to enter apartment characteristics and explore predicted prices, prediction intervals, and similar properties.

## Model Performance

Several regression models were evaluated to compare predictive performance.

| Model | Test MAE (RON) | Test RMSE (RON) | Test R² |
|---|---:|---:|---:|
| Dummy Regressor | 181,380 | 275,561 | 0.000 |
| Linear Regression | 98,700 | 162,136 | 0.654 |
| Random Forest | 90,030 | 158,441 | 0.669 |
| Tuned XGBoost | 91,657 | 161,663 | 0.656 |

The tuned XGBoost model was selected based on cross-validation performance, rather than choosing the model with the lowest test-set error.

**Best XGBoost cross-validation MAE:** approximately 88,139 RON.

The split conformal prediction approach achieved **88.01% empirical coverage** on the test set, compared with its nominal target of 90%.

These results reflect the available historical transaction data and should not be interpreted as guaranteed market valuations.

## Dataset

The project uses publicly available real estate transaction data from **Bucharest, Sector 2, Romania (2022)**, obtained from the Romanian government's open data portal, [data.gov.ro](https://data.gov.ro/).

The original dataset contains **4,325 transactions**, including **3,987 apartment transactions**.

After data cleaning and filtering, the final dataset contains **3,584 apartment transactions**.

### Data Source & License

**Dataset:** [Lista tranzactiilor imobiliare pe raza sectorului 2 in anul 2022](https://data.gov.ro/dataset/lista-tranzactiilor-imobiliare-pe-raza-sectorului-2-in-anul-2022)

**Publisher:** Primăria Sectorului 2 București

**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The original dataset is publicly available through Romania's official open data portal, data.gov.ro.

The data was cleaned, filtered, and transformed for machine learning purposes. The original data publisher is credited in accordance with the CC BY 4.0 license.

This project is an independent educational and portfolio project and is not affiliated with the data publisher.

### Data Preparation

The preprocessing pipeline includes:

- Filtering apartment transactions from the original dataset.
- Standardizing column names and neighborhood labels.
- Handling missing values and invalid construction years.
- Removing records without usable area or transaction price.
- Applying exploratory filtering thresholds of 10 m² minimum usable area and 2,000 RON/m² minimum transaction price.
- Extracting the transaction quarter.
- Validating the processed dataset before modeling.

The preprocessing steps can be reproduced using `src/data/prepare_data.py` and `src/data/validate_data.py`.

## Methodology

The modeling workflow includes an 80/20 train-test split, numerical preprocessing, categorical encoding, model comparison, and cross-validation.

The models use apartment characteristics such as usable area, construction year, neighborhood, street type, and transaction quarter.

Hyperparameter tuning was performed using `RandomizedSearchCV`.

SHAP was used for model interpretation, while split conformal prediction was used to estimate prediction uncertainty.

A nearest-neighbors model was implemented to retrieve similar apartments based on preprocessed property characteristics.

## Limitations

- The dataset covers only Bucharest's Sector 2 and transactions from 2022, so the results may not generalize to other locations or current market conditions.
- Important property characteristics, such as floor number, apartment condition, and number of rooms, are unavailable.
- Filtering thresholds are exploratory choices and may introduce selection bias.
- The dataset does not provide unique transaction identifiers, making it difficult to distinguish identical transactions from duplicate records.
- Prediction intervals do not guarantee coverage for every individual apartment.
- The model is intended for educational and analytical purposes, not professional property appraisal.

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/arianarobu/real-estate-intelligence.git
cd real-estate-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare and validate the data

Place the original transaction CSV in `data/raw/` using the filename expected by the preprocessing script.

Then run:

```bash
python src/data/prepare_data.py
python src/data/validate_data.py
```

### 5. Launch the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

The application will open in your browser and allow you to predict apartment prices, inspect prediction intervals, and explore similar properties.

The trained models are stored in the `models/` directory.


## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- FastAPI
- Docker

