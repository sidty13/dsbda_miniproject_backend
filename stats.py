import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from database import get_db


def _load_df(start: str = None, end: str = None) -> pd.DataFrame:
    """Load gold_prices from DB into a DataFrame, optionally filtered by date range."""
    conn = get_db()
    if start and end:
        df = pd.read_sql_query(
            "SELECT * FROM gold_prices WHERE date BETWEEN ? AND ? ORDER BY date",
            conn, params=(start, end)
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM gold_prices ORDER BY date", conn
        )
    conn.close()
    return df


def get_statistics(start: str = None, end: str = None) -> dict:
    """
    Compute descriptive statistics for OHLCV columns.
    Covers: mean, median, mode, std, variance, min, max,
            quartiles, IQR, skewness, kurtosis, coefficient of variation.
    """
    df = _load_df(start, end)

    if df.empty:
        return {"error": "No data found for the given range."}

    numeric_cols = ["open", "high", "low", "close", "volume", "daily_return", "price_range"]
    result = {
        "total_records": len(df),
        "date_range": {
            "from": df["date"].iloc[0],
            "to":   df["date"].iloc[-1],
        },
        "columns": {}
    }

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1  = float(np.percentile(series, 25))
        q3  = float(np.percentile(series, 75))
        iqr = round(q3 - q1, 4)

        mode_result = scipy_stats.mode(series.round(2), keepdims=True)
        mode_val    = round(float(mode_result.mode[0]), 4)

        result["columns"][col] = {
            "mean":                   round(float(series.mean()), 4),
            "median":                 round(float(series.median()), 4),
            "mode":                   mode_val,
            "std_deviation":          round(float(series.std()), 4),
            "variance":               round(float(series.var()), 4),
            "min":                    round(float(series.min()), 4),
            "max":                    round(float(series.max()), 4),
            "range":                  round(float(series.max() - series.min()), 4),
            "q1_25th_percentile":     round(q1, 4),
            "q2_50th_percentile":     round(float(np.percentile(series, 50)), 4),
            "q3_75th_percentile":     round(q3, 4),
            "iqr":                    iqr,
            "skewness":               round(float(series.skew()), 4),
            "kurtosis":               round(float(series.kurtosis()), 4),
            "coefficient_of_variation": round(
                float(series.std() / series.mean() * 100), 4
            ) if series.mean() != 0 else None,
            "count": int(series.count()),
        }

    # Interpretation notes for close price (most relevant column)
    close = df["close"].dropna()
    skew  = float(close.skew())
    kurt  = float(close.kurtosis())

    result["interpretation"] = {
        "close_skewness": (
            "positively skewed (long right tail — occasional price spikes)"
            if skew > 0.5 else
            "negatively skewed (long left tail — occasional price drops)"
            if skew < -0.5 else
            "approximately symmetric"
        ),
        "close_kurtosis": (
            "leptokurtic (heavy tails — more extreme price movements than normal)"
            if kurt > 1 else
            "platykurtic (thin tails — fewer extreme price movements)"
            if kurt < -1 else
            "mesokurtic (approximately normal distribution)"
        ),
    }

    return result