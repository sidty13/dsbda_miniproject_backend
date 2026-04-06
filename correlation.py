import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from database import get_db


def _load_df(start: str = None, end: str = None) -> pd.DataFrame:
    conn = get_db()
    if start and end:
        df = pd.read_sql_query(
            "SELECT * FROM gold_prices WHERE date BETWEEN ? AND ? ORDER BY date",
            conn, params=(start, end)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM gold_prices ORDER BY date", conn)
    conn.close()
    return df


def _interpret_correlation(r: float) -> str:
    """Human-readable label for a Pearson r value."""
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak / negligible"
    return f"{strength} {direction} correlation"


def get_correlation(start: str = None, end: str = None) -> dict:
    """
    Compute:
    - Pearson correlation matrix for all numeric columns
    - Spearman correlation matrix (rank-based, handles non-linearity)
    - Pairwise p-values (statistical significance)
    - Top correlated and least correlated column pairs
    - Interpretation for each pair
    """
    df = _load_df(start, end)

    if df.empty:
        return {"error": "No data found for the given range."}

    numeric_cols = ["open", "high", "low", "close", "volume", "price_range", "daily_return"]
    df_num = df[numeric_cols].dropna()

    # ── Pearson Correlation ───────────────────────────────────────────────────
    pearson_matrix = df_num.corr(method="pearson").round(4)

    # ── Spearman Correlation ──────────────────────────────────────────────────
    spearman_matrix = df_num.corr(method="spearman").round(4)

    # ── P-values for Pearson ──────────────────────────────────────────────────
    pvalue_matrix = pd.DataFrame(
        np.ones((len(numeric_cols), len(numeric_cols))),
        index=numeric_cols, columns=numeric_cols
    )
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                _, p = scipy_stats.pearsonr(df_num[col1], df_num[col2])
                pvalue_matrix.loc[col1, col2] = round(p, 6)

    # ── Pairwise Analysis ─────────────────────────────────────────────────────
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            r  = float(pearson_matrix.loc[col1, col2])
            rho = float(spearman_matrix.loc[col1, col2])
            p  = float(pvalue_matrix.loc[col1, col2])
            pairs.append({
                "column_1":            col1,
                "column_2":            col2,
                "pearson_r":           round(r, 4),
                "spearman_rho":        round(rho, 4),
                "p_value":             round(p, 6),
                "statistically_significant": p < 0.05,
                "interpretation":      _interpret_correlation(r),
            })

    pairs_sorted = sorted(pairs, key=lambda x: abs(x["pearson_r"]), reverse=True)

    # ── Key Findings ──────────────────────────────────────────────────────────
    close_corrs = [
        p for p in pairs if p["column_1"] == "close" or p["column_2"] == "close"
    ]
    close_corrs_sorted = sorted(close_corrs, key=lambda x: abs(x["pearson_r"]), reverse=True)

    return {
        "total_records_used": len(df_num),
        "date_range": {
            "from": df["date"].iloc[0],
            "to":   df["date"].iloc[-1],
        },

        # Full matrices as nested dicts
        "pearson_correlation_matrix":   pearson_matrix.to_dict(),
        "spearman_correlation_matrix":  spearman_matrix.to_dict(),
        "p_value_matrix":               pvalue_matrix.round(6).to_dict(),

        # All unique pairs ranked by |r|
        "pairwise_analysis":            pairs_sorted,

        # Findings focused on Close price
        "close_price_correlations":     close_corrs_sorted,

        "key_insights": {
            "strongest_pair": {
                "pair":  f"{pairs_sorted[0]['column_1']} ↔ {pairs_sorted[0]['column_2']}",
                "r":     pairs_sorted[0]["pearson_r"],
                "label": pairs_sorted[0]["interpretation"],
            },
            "weakest_pair": {
                "pair":  f"{pairs_sorted[-1]['column_1']} ↔ {pairs_sorted[-1]['column_2']}",
                "r":     pairs_sorted[-1]["pearson_r"],
                "label": pairs_sorted[-1]["interpretation"],
            },
            "significant_pairs_count": sum(1 for p in pairs if p["statistically_significant"]),
            "total_pairs":             len(pairs),
            "note": (
                "Open, High, Low, and Close are expected to be highly correlated "
                "as they all track the same underlying gold price. "
                "Volume and daily_return correlations are more analytically interesting."
            ),
        },
    }