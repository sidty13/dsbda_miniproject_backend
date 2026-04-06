import pandas as pd
import numpy as np
from database import get_db


def _load_df() -> pd.DataFrame:
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM gold_prices ORDER BY date", conn)
    conn.close()
    return df


def get_preprocessing_info() -> dict:
    """
    Analyse the dataset for:
    - Missing values / nulls
    - Duplicate records
    - Outliers (IQR method + Z-score method)
    - Data types
    - Value ranges and sanity checks
    - Zero-volume trading days
    - Data quality score
    """
    df = _load_df()
    numeric_cols = ["open", "high", "low", "close", "volume", "daily_return", "price_range"]
    result = {}

    # ── 1. Basic Info ─────────────────────────────────────────────────────────
    result["dataset_info"] = {
        "total_records":  len(df),
        "total_columns":  len(df.columns),
        "columns":        list(df.columns),
        "date_range": {
            "from": df["date"].iloc[0],
            "to":   df["date"].iloc[-1],
        },
        "data_types": {col: str(df[col].dtype) for col in df.columns},
    }

    # ── 2. Missing Values ─────────────────────────────────────────────────────
    null_counts = df[numeric_cols].isnull().sum()
    result["missing_values"] = {
        col: {
            "null_count":   int(null_counts[col]),
            "null_percent": round(float(null_counts[col] / len(df) * 100), 4),
            "status":       "clean" if null_counts[col] == 0 else "has nulls",
        }
        for col in numeric_cols
    }
    result["missing_values"]["summary"] = {
        "total_nulls":       int(null_counts.sum()),
        "columns_with_nulls": int((null_counts > 0).sum()),
    }

    # ── 3. Duplicate Records ──────────────────────────────────────────────────
    dup_count = int(df.duplicated(subset=["date"]).sum())
    result["duplicates"] = {
        "duplicate_date_count": dup_count,
        "status": "clean" if dup_count == 0 else f"{dup_count} duplicate dates found",
    }

    # ── 4. Outliers ───────────────────────────────────────────────────────────
    outlier_cols = ["open", "high", "low", "close", "volume", "price_range"]
    result["outliers"] = {}

    for col in outlier_cols:
        series = df[col].dropna()

        # IQR method
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_outliers = int(((series < lower_iqr) | (series > upper_iqr)).sum())

        # Z-score method (|z| > 3)
        z_scores    = np.abs((series - series.mean()) / series.std())
        z_outliers  = int((z_scores > 3).sum())

        result["outliers"][col] = {
            "iqr_method": {
                "lower_fence":    round(float(lower_iqr), 4),
                "upper_fence":    round(float(upper_iqr), 4),
                "outlier_count":  iqr_outliers,
                "outlier_percent": round(iqr_outliers / len(series) * 100, 4),
            },
            "zscore_method": {
                "threshold":      3,
                "outlier_count":  z_outliers,
                "outlier_percent": round(z_outliers / len(series) * 100, 4),
            },
        }

    # ── 5. Sanity Checks ──────────────────────────────────────────────────────
    # High should always >= Low, Close should be between High and Low
    high_lt_low      = int((df["high"] < df["low"]).sum())
    close_above_high = int((df["close"] > df["high"]).sum())
    close_below_low  = int((df["close"] < df["low"]).sum())
    negative_prices  = int((df[["open", "high", "low", "close"]] < 0).any(axis=1).sum())
    negative_volume  = int((df["volume"] < 0).sum())

    result["sanity_checks"] = {
        "high_less_than_low":      {"count": high_lt_low,      "status": "clean" if high_lt_low == 0      else "FAIL"},
        "close_above_high":        {"count": close_above_high, "status": "clean" if close_above_high == 0 else "FAIL"},
        "close_below_low":         {"count": close_below_low,  "status": "clean" if close_below_low == 0  else "FAIL"},
        "negative_prices":         {"count": negative_prices,  "status": "clean" if negative_prices == 0  else "FAIL"},
        "negative_volume":         {"count": negative_volume,  "status": "clean" if negative_volume == 0  else "FAIL"},
        "note": (
            "Violations in close_above_high / close_below_low originate from the "
            "raw Kaggle CSV and reflect upstream data quality issues, not pipeline errors."
        ),
    }

    # ── 6. Zero Volume Days ───────────────────────────────────────────────────
    zero_vol = df[df["volume"] == 0]
    result["zero_volume_days"] = {
        "count":   int(len(zero_vol)),
        "percent": round(len(zero_vol) / len(df) * 100, 4),
        "note":    "Zero volume days may indicate illiquid trading sessions or data gaps.",
        "sample_dates": zero_vol["date"].head(5).tolist(),
    }

    # ── 7. Value Ranges ───────────────────────────────────────────────────────
    result["value_ranges"] = {
        col: {
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }
        for col in outlier_cols
    }

    # ── 8. Data Quality Score (simple heuristic) ──────────────────────────────
    total_checks = 5
    passed = sum([
        high_lt_low == 0,
        close_above_high == 0,
        close_below_low == 0,
        negative_prices == 0,
        negative_volume == 0,
    ])
    null_penalty  = min(result["missing_values"]["summary"]["total_nulls"] / len(df) * 100, 20)
    quality_score = round((passed / total_checks * 80) + (20 - null_penalty), 2)

    result["data_quality"] = {
        "score":            quality_score,
        "out_of":           100,
        "sanity_checks_passed": f"{passed}/{total_checks}",
        "grade": (
            "A" if quality_score >= 90 else
            "B" if quality_score >= 75 else
            "C" if quality_score >= 60 else "D"
        ),
    }

    return result