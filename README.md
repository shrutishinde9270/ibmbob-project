# 🚗 Car Sell Price Prediction

A full-stack machine learning web application built entirely in Python that predicts used car selling prices using a Random Forest Regressor, with an interactive Streamlit frontend.

---

## 📁 Project Structure

```
car_data_analysis/
├── app.py                      # Streamlit frontend — main entry point
├── requirements.txt            # Python dependencies
├── Car Sell Dataset.csv        # Source dataset (999 records, 12 columns)
├── README.md                   # This file
├── PROJECT_REPORT.md           # Detailed project report
├── model/
│   ├── train.py                # Data preprocessing + model training script
│   ├── car_price_model.pkl     # Trained Random Forest model (auto-generated)
│   └── encoders.pkl            # Label encoders for categorical features (auto-generated)
└── utils/
    └── preprocess.py           # Shared feature engineering and encoding logic
```

---

## 🧰 Tech Stack

| Component       | Library / Tool              |
|-----------------|-----------------------------|
| Frontend UI     | Streamlit                   |
| ML Model        | scikit-learn (RandomForest) |
| Data Processing | pandas, numpy               |
| Visualizations  | Plotly, Matplotlib, Seaborn |
| Model Saving    | joblib                      |
| Language        | Python 3.8+                 |

---

## 📊 Dataset

**File:** `Car Sell Dataset.csv`  
**Records:** 999  
**Columns:** 12

| Column         | Type        | Description                          |
|----------------|-------------|--------------------------------------|
| Brand          | Categorical | Car manufacturer (e.g. Hyundai)      |
| Model Name     | Categorical | Car model (e.g. Grand i10)           |
| Model Variant  | Categorical | Trim/variant (e.g. Magna)            |
| Car Type       | Categorical | SUV, Sedan, Hatchback, etc.          |
| Transmission   | Categorical | Manual / Automatic                   |
| Fuel Type      | Categorical | Petrol, Diesel, CNG, Electric, etc.  |
| Year           | Numeric     | Manufacturing year                   |
| Kilometers     | Numeric     | Total kilometers driven              |
| Owner          | Categorical | 1st, 2nd, 3rd+ owner                 |
| State          | Categorical | Indian state of registration         |
| Accidental     | Categorical | Yes / No                             |
| Price          | Numeric     | Selling price in INR (target)        |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model/train.py
```

This will:
- Load and preprocess `Car Sell Dataset.csv`
- Engineer the `Car Age` feature from `Year`
- Label-encode all categorical columns
- Train a `RandomForestRegressor` (200 trees, max depth 20)
- Evaluate on 20% held-out test set
- Save `model/car_price_model.pkl` and `model/encoders.pkl`

### 3. Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Note:** You can also train the model directly from the **🤖 Train Model** page inside the app without running step 2 manually.

---

## 🖥️ Application Pages

### 🏠 Home
- Dataset overview with key statistics (total records, unique brands, average price, price range)
- Scrollable dataset preview (first 20 rows)
- Column info table (types, nulls, unique values)
- Price distribution histogram

### 📊 Exploratory Analysis
Five analysis tabs:

| Tab                  | Charts                                                   |
|----------------------|----------------------------------------------------------|
| Brand Analysis       | Top brands by listings & avg price; Price by Car Type    |
| Fuel & Transmission  | Pie charts of distribution; Box plots by price           |
| Year & Age           | Avg price trend by year; Cars per year; Price by owner   |
| Mileage vs Price     | Scatter plot (km vs price); Accidental price comparison  |
| State Analysis       | Avg price by state; Top 15 states by listings            |

### 🤖 Train Model
- Model configuration summary
- One-click training trigger with live progress spinner
- Displays MAE, RMSE, and R² score after training
- Feature importance bar chart

### 🔮 Predict Price
- Smart cascading dropdowns (Brand → Model Name → Model Variant)
- Sliders and inputs for Year, Kilometers, Owner, State, Accidental
- Predicted price displayed prominently
- Similar cars from the dataset shown for comparison
- Bar chart comparing prediction against similar listings

---

## 🤖 ML Pipeline

```
Raw CSV
  └─► load_data()          — pandas read_csv
  └─► add_car_age()        — Car Age = Current Year - Year
  └─► encode_categoricals() — LabelEncoder per categorical column
  └─► train_test_split()   — 80% train / 20% test, random_state=42
  └─► RandomForestRegressor(n_estimators=200, max_depth=20)
  └─► Evaluate (MAE, RMSE, R²)
  └─► joblib.dump()        — save model + encoders
```

**Features used (11):**
`Brand`, `Model Name`, `Model Variant`, `Car Type`, `Transmission`, `Fuel Type`, `Car Age`, `Kilometers`, `Owner`, `State`, `Accidental`

**Target:** `Price` (INR)

---

## 📈 Model Performance

| Metric | Value        |
|--------|--------------|
| MAE    | ₹3,36,196    |
| RMSE   | ₹4,08,742    |
| R²     | 0.27         |

---

## 🔄 Retraining

To retrain with fresh data, either:
- Replace `Car Sell Dataset.csv` with updated data and run `python model/train.py`
- Or use the **🤖 Train Model** page in the app and click **Train Model**

---

## 📄 License

This project is for educational and analytical purposes.
