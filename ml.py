import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from database import get_db


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw OHLCV data."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    df["date_ordinal"] = pd.to_datetime(df["date"]).map(datetime.toordinal)

    # Rolling averages (use shift to avoid leakage)
    df["ma_5"]  = df["close"].shift(1).rolling(5).mean()
    df["ma_20"] = df["close"].shift(1).rolling(20).mean()

    # Lag features
    df["lag_1"] = df["close"].shift(1)
    df["lag_3"] = df["close"].shift(3)
    df["lag_5"] = df["close"].shift(5)

    # Volatility
    df["volatility_5"] = df["close"].shift(1).rolling(5).std()

    return df


FEATURE_COLS = [
    "date_ordinal", "open", "high", "low", "volume",
    "price_range", "ma_5", "ma_20", "lag_1", "lag_3", "lag_5", "volatility_5",
]


def train_model():
    """Train GradientBoostingRegressor + LinearRegression on all historical data."""
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume, price_range FROM gold_prices ORDER BY date",
        conn
    )
    conn.close()

    if len(df) < 50:
        return None, None, None, {}

    df = _build_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].values
    y = df["close"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, shuffle=False
    )

    # Primary model — Gradient Boosting (tuned for time-series drift)
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.7,
        random_state=42,
    )
    gbr.fit(X_train, y_train)

    # Fallback model — Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Metrics on test split
    gbr_preds = gbr.predict(X_test)
    lr_preds  = lr.predict(X_test)

    metrics = {
        "gbr": {
            "mae":  round(float(mean_absolute_error(y_test, gbr_preds)), 4),
            "r2":   round(float(r2_score(y_test, gbr_preds)), 4),
        },
        "lr": {
            "mae":  round(float(mean_absolute_error(y_test, lr_preds)), 4),
            "r2":   round(float(r2_score(y_test, lr_preds)), 4),
        },
        "train_rows": len(X_train),
        "test_rows":  len(X_test),
    }

    print(f"[ML] GBR  → MAE: {metrics['gbr']['mae']}, R²: {metrics['gbr']['r2']}")
    print(f"[ML] LR   → MAE: {metrics['lr']['mae']}, R²: {metrics['lr']['r2']}")

    return gbr, lr, scaler, metrics


def predict_close(
    gbr,
    lr,
    scaler,
    record: dict,
    all_records: list[dict],
) -> dict:
    """
    Predict close price for a single record using both models.
    all_records must be sorted by date and include the record.
    Returns dict with gbr_predicted and lr_predicted.
    """
    if gbr is None or scaler is None:
        return {"gbr_predicted": None, "lr_predicted": None}

    df = pd.DataFrame(all_records)
    df = _build_features(df)

    row = df[df["date"] == record["date"]]
    if row.empty or row[FEATURE_COLS].isnull().values.any():
        return {"gbr_predicted": None, "lr_predicted": None}

    X = row[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    return {
        "gbr_predicted": round(float(gbr.predict(X_scaled)[0]), 2),
        "lr_predicted":  round(float(lr.predict(X_scaled)[0]), 2),
    }