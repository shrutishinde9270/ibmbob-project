# Car Sell Price Prediction — Project Report

**Project Title:** Used Car Selling Price Prediction using Machine Learning  
**Technology:** Python (Streamlit + scikit-learn)  
**Dataset:** Car Sell Dataset (999 records, 12 features)  
**Model:** Random Forest Regressor  

---

## 1. Introduction

The used car market in India is one of the largest in the world, with millions of transactions happening annually. Pricing a used car accurately is a complex problem — it depends on brand reputation, model variant, fuel type, kilometers driven, ownership history, geographic location, accident history, and the age of the vehicle.

This project builds an end-to-end machine learning application that:
- Analyses the used car dataset through rich visualizations
- Trains a Random Forest regression model to predict selling prices
- Exposes predictions through an interactive Streamlit web application

The entire solution — data processing, model training, and frontend — is written in Python.

---

## 2. Objectives

1. Perform exploratory data analysis (EDA) to understand price-influencing factors
2. Engineer meaningful features from raw data
3. Build and evaluate a regression model for price prediction
4. Deliver an interactive web application for end-user price prediction

---

## 3. Dataset Description

### Source
`Car Sell Dataset.csv` — a curated dataset of 999 used car listings in India.

### Schema

| # | Column        | Data Type   | Description                              |
|---|---------------|-------------|------------------------------------------|
| 1 | Brand         | Categorical | Manufacturer name (e.g. Maruti Suzuki)   |
| 2 | Model Name    | Categorical | Car model name (e.g. Swift)              |
| 3 | Model Variant | Categorical | Trim level (e.g. VXI, ZXI)              |
| 4 | Car Type      | Categorical | Body style: SUV, Sedan, Hatchback, etc.  |
| 5 | Transmission  | Categorical | Manual or Automatic                      |
| 6 | Fuel Type     | Categorical | Petrol, Diesel, CNG, Electric, etc.      |
| 7 | Year          | Numeric     | Year of manufacture                      |
| 8 | Kilometers    | Numeric     | Odometer reading                         |
| 9 | Owner         | Categorical | Ownership order: 1st, 2nd, 3rd+          |
| 10| State         | Categorical | State of registration in India           |
| 11| Accidental    | Categorical | Whether car has accident history (Yes/No)|
| 12| Price         | Numeric     | Selling price in INR (target variable)   |

### Summary Statistics
- **Total records:** 999
- **Missing values:** None (clean dataset)
- **Price range:** Highly variable (budget hatchbacks to premium SUVs)
- **Year range:** 1990–2025
- **Unique brands:** Multiple major Indian and international manufacturers

---

## 4. Exploratory Data Analysis

### 4.1 Price Distribution
The selling price distribution is right-skewed, with most cars priced in the low-to-mid range and a long tail for premium/luxury vehicles.

### 4.2 Brand Analysis
- Maruti Suzuki dominates listings by volume, reflecting its market share
- Premium brands (MG, Jeep, Kia) command higher average prices
- Entry-level brands (Alto, Ignis, WagonR) have lower average prices

### 4.3 Fuel Type Analysis
- Petrol and Diesel are the most common fuel types
- Electric vehicles command premium prices despite being newer entries
- CNG vehicles tend to price lower due to limited demand

### 4.4 Transmission
- Manual transmission cars far outnumber automatics
- Automatic cars show higher average prices, likely due to correlation with premium segments

### 4.5 Year & Age Impact
- Newer cars (2019–2024) command significantly higher prices
- A clear negative correlation exists between car age and price
- Cars from 2000–2010 are clustered in the budget segment

### 4.6 Ownership History
- 1st owner cars are priced highest
- Price drops progressively with each ownership change
- 3rd+ owner vehicles occupy the lowest price band

### 4.7 Mileage vs Price
- Higher kilometer readings correlate with lower prices
- Electric and low-mileage cars cluster in the high-price zone
- The scatter is significant, suggesting price is multi-factorial

### 4.8 Accidental Cars
- Non-accidental cars price consistently higher
- Accident history causes a measurable price depreciation

---

## 5. Feature Engineering

| Feature     | Source        | Transformation                        |
|-------------|---------------|---------------------------------------|
| Car Age     | Year          | `Car Age = Current Year - Year`       |
| Brand       | Brand         | Label Encoded                         |
| Model Name  | Model Name    | Label Encoded                         |
| Model Variant| Model Variant| Label Encoded                         |
| Car Type    | Car Type      | Label Encoded                         |
| Transmission| Transmission  | Label Encoded                         |
| Fuel Type   | Fuel Type     | Label Encoded                         |
| Owner       | Owner         | Label Encoded                         |
| State       | State         | Label Encoded                         |
| Accidental  | Accidental    | Label Encoded                         |
| Kilometers  | Kilometers    | Used as-is (numeric)                  |

**Total features fed to model: 11**  
**Target variable: Price (INR)**

Label encoding was chosen over one-hot encoding to keep dimensionality low given the dataset size (999 rows), and because Random Forest handles ordinal-like encoded categoricals well.

---

## 6. Model Architecture

### Algorithm: Random Forest Regressor

Random Forest was chosen for the following reasons:
- Handles mixed feature types (categorical + numeric) after label encoding
- Robust to outliers in the price distribution
- Provides built-in feature importance
- No need for feature scaling
- Handles interactions between features automatically

### Hyperparameters

| Parameter         | Value | Rationale                                      |
|-------------------|-------|------------------------------------------------|
| n_estimators      | 200   | More trees = lower variance, stable predictions|
| max_depth         | 20    | Deep enough to capture non-linear interactions |
| min_samples_split | 4     | Prevents overfitting on small leaf nodes        |
| min_samples_leaf  | 2     | Ensures each leaf has sufficient samples        |
| random_state      | 42    | Reproducibility                                |
| n_jobs            | -1    | Use all CPU cores for faster training           |

### Train / Test Split

| Split | Size | Percentage |
|-------|------|------------|
| Train | 799  | 80%        |
| Test  | 200  | 20%        |

`random_state=42` ensures reproducible splits.

---

## 7. Model Evaluation

### Metrics

| Metric                    | Value       | Interpretation                                      |
|---------------------------|-------------|-----------------------------------------------------|
| MAE (Mean Absolute Error) | ₹3,36,196   | Average prediction error of ~₹3.36 lakh             |
| RMSE (Root Mean Sq. Error)| ₹4,08,742   | Penalises large errors; ~₹4.09 lakh                 |
| R² Score                  | 0.27        | Model explains ~27% of price variance               |

### Analysis
The R² of 0.27 indicates moderate predictive power. This is expected given:
- Only 999 training samples — a limited dataset for a high-variance regression task
- High natural price variance in used car markets (same model, same year can have vastly different prices)
- Label encoding loses ordinal relationships for multi-class categoricals
- No external market data (city pricing, demand trends) is included

**Potential improvements for higher accuracy:**
- Larger dataset (10,000+ records)
- Target encoding or embedding for high-cardinality categoricals (Model Variant has many levels)
- Gradient Boosting (XGBoost / LightGBM)
- Log-transform the target price to reduce skew
- Add interaction features (Brand × Fuel Type, Age × Kilometers)

---

## 8. Application Architecture

```
┌─────────────────────────────────────────────────────┐
│                   app.py (Streamlit)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Home    │ │   EDA    │ │  Train   │ │Predict │ │
│  │  Page    │ │  Page    │ │  Page    │ │  Page  │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │ imports
          ┌────────────┴────────────┐
          │                         │
┌─────────▼──────────┐   ┌──────────▼──────────┐
│  utils/preprocess  │   │   model/train.py     │
│  .py               │   │                      │
│  - load_data()     │   │  - train_and_save()  │
│  - add_car_age()   │   │  - load_model()      │
│  - encode_categ.() │   │                      │
│  - encode_single() │   └──────────┬───────────┘
└────────────────────┘              │ saves
                           ┌────────▼────────┐
                           │  model/          │
                           │  ├ *.pkl (model) │
                           │  └ *.pkl (encs.) │
                           └─────────────────┘
```

### Data Flow for Prediction

```
User Input (form)
  └─► encode_single_input()    — apply same encoders used in training
  └─► model.predict()          — Random Forest inference
  └─► Display price + chart    — Streamlit UI
```

---

## 9. File Reference

| File                      | Role                                                    | Lines |
|---------------------------|---------------------------------------------------------|-------|
| `app.py`                  | Streamlit 4-page application, all UI and chart logic    | ~350  |
| `utils/preprocess.py`     | Feature engineering, encoding, data loading utilities   | ~88   |
| `model/train.py`          | Model training, evaluation, persistence                 | ~79   |
| `requirements.txt`        | Python package dependencies                             | 8     |
| `Car Sell Dataset.csv`    | Raw dataset                                             | 1000  |
| `model/car_price_model.pkl`| Serialised trained Random Forest model                 | —     |
| `model/encoders.pkl`      | Serialised LabelEncoder objects per categorical column  | —     |

---

## 10. How to Run

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Train the model (generates .pkl files)
python model/train.py

# 3. Launch the web app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

Alternatively, skip step 2 and use the **🤖 Train Model** page inside the app to train with one click.

---

## 11. Conclusion

This project successfully delivers an end-to-end used car price prediction system:

- **EDA** reveals that car age, brand, fuel type, and ownership history are the strongest price drivers
- **Random Forest** provides a baseline prediction model suitable for this dataset size
- **Streamlit** enables rapid delivery of a clean, multi-page interactive UI in pure Python
- The modular architecture (`utils/`, `model/`, `app.py`) makes the project easy to extend

Future work could improve the model by expanding the dataset, incorporating external market signals, and experimenting with gradient boosting algorithms such as XGBoost or LightGBM.

---

*Report generated for the Car Sell Price Prediction project.*
