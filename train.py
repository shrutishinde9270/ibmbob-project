import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_data, preprocess, get_features_and_target

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Car Sell Dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_price_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoders.pkl")


def train_and_save():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")

    print("Preprocessing...")
    df_processed, encoders = preprocess(df, fit=True)

    X, y = get_features_and_target(df_processed)
    print(f"  Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n[Model Evaluation]")
    print(f"  MAE  : Rs.{mae:,.0f}")
    print(f"  RMSE : Rs.{rmse:,.0f}")
    print(f"  R2   : {r2:.4f}")

    # Save model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\n[OK] Model saved to: {MODEL_PATH}")
    print(f"[OK] Encoders saved to: {ENCODERS_PATH}")

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, encoders, metrics


def load_model():
    """Load saved model and encoders."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(
            "Model not found. Please train the model first via the 'Train Model' page."
        )
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


if __name__ == "__main__":
    train_and_save()
