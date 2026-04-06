from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from database import init_db, get_db
from models import GoldRecord, GoldRangeResponse, RangeSummary
from ml import train_model, predict_close


# ── App state ─────────────────────────────────────────────────────────────────

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Init DB and load CSV on first run
    init_db()

    # 2. Train ML models
    gbr, lr, scaler, metrics = train_model()
    _state["gbr"]     = gbr
    _state["lr"]      = lr
    _state["scaler"]  = scaler
    _state["metrics"] = metrics
    print(f"[App] Startup complete. Model metrics: {metrics}")
    yield


app = FastAPI(
    title="Gold Price Dynamics API",
    description=(
        "Fetch and predict gold (XAU/USD) prices using historical data (2000–2026). "
        "Two routes: single date and date range. "
        "Predictions via GradientBoostingRegressor and LinearRegression (scikit-learn)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> str:
    """Validate and return date as YYYY-MM-DD string."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD"
        )


def _row_to_dict(row) -> dict:
    return dict(row)


def _fetch_context_records(date: str) -> list[dict]:
    """Fetch enough historical rows to compute lag/rolling features."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT date, open, high, low, close, volume, price_range
        FROM gold_prices
        WHERE date <= ?
        ORDER BY date DESC
        LIMIT 30
        """,
        (date,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


# ── Route 1: Single Date ──────────────────────────────────────────────────────

@app.get(
    "/gold/date/{date}",
    response_model=GoldRecord,
    summary="Get gold price for a single date",
    tags=["Gold Prices"],
)
def get_gold_by_date(date: str):
    """
    Returns OHLCV data + engineered features + ML predictions for a single date.

    - **date**: format `YYYY-MM-DD`
    - **predicted_close**: average of GBR and LR model predictions
    """
    parsed = _parse_date(date)

    conn = get_db()
    row = conn.execute(
        "SELECT * FROM gold_prices WHERE date = ?", (parsed,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for date: {parsed}. "
                   "Market may have been closed (weekend/holiday)."
        )

    record = _row_to_dict(row)
    context = _fetch_context_records(parsed)

    preds = predict_close(
        _state["gbr"], _state["lr"], _state["scaler"],
        record, context
    )

    # LR generalises better on this dataset (R²=0.9998); GBR kept as reference
    gbr_p = preds["gbr_predicted"]
    lr_p  = preds["lr_predicted"]
    predicted_close = lr_p if lr_p is not None else gbr_p

    return GoldRecord(
        date=record["date"],
        open=record["open"],
        high=record["high"],
        low=record["low"],
        close=record["close"],
        volume=record["volume"],
        daily_return=record["daily_return"],
        price_range=record["price_range"],
        predicted_close=predicted_close,
    )


# ── Route 2: Date Range ───────────────────────────────────────────────────────

@app.get(
    "/gold/range",
    response_model=GoldRangeResponse,
    summary="Get gold prices for a date range",
    tags=["Gold Prices"],
)
def get_gold_by_range(
    start: str = Query(..., description="Start date (YYYY-MM-DD)", examples=["2024-01-01"]),
    end:   str = Query(..., description="End date   (YYYY-MM-DD)", examples=["2024-03-31"]),
):
    """
    Returns OHLCV data for every trading day in the range, plus:

    - Per-record ML predictions (GBR + LR averaged)
    - Aggregated summary: avg/max/min close, total volume, % price change
    """
    start_date = _parse_date(start)
    end_date   = _parse_date(end)

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="'start' must be before or equal to 'end'."
        )

    # Fetch 30 rows before start for lag/rolling feature computation
    conn = get_db()
    context_rows = conn.execute(
        """
        SELECT date, open, high, low, close, volume, price_range
        FROM gold_prices
        WHERE date < ?
        ORDER BY date DESC
        LIMIT 30
        """,
        (start_date,),
    ).fetchall()

    range_rows = conn.execute(
        """
        SELECT * FROM gold_prices
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        (start_date, end_date),
    ).fetchall()
    conn.close()

    if not range_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No trading data found between {start_date} and {end_date}."
        )

    context  = [dict(r) for r in reversed(context_rows)]
    all_recs = context + [dict(r) for r in range_rows]

    results = []
    for row in range_rows:
        record = dict(row)
        preds  = predict_close(
            _state["gbr"], _state["lr"], _state["scaler"],
            record, all_recs
        )
        gbr_p = preds["gbr_predicted"]
        lr_p  = preds["lr_predicted"]
        predicted_close = lr_p if lr_p is not None else gbr_p

        results.append(GoldRecord(
            date=record["date"],
            open=record["open"],
            high=record["high"],
            low=record["low"],
            close=record["close"],
            volume=record["volume"],
            daily_return=record["daily_return"],
            price_range=record["price_range"],
            predicted_close=predicted_close,
        ))

    closes  = [r.close for r in results]
    returns = [r.daily_return for r in results if r.daily_return is not None]

    price_change_pct = round(
        (closes[-1] - closes[0]) / closes[0] * 100, 4
    ) if len(closes) >= 2 else 0.0

    summary = RangeSummary(
        avg_close=round(sum(closes) / len(closes), 2),
        max_close=round(max(closes), 2),
        min_close=round(min(closes), 2),
        avg_daily_return=round(sum(returns) / len(returns), 4) if returns else None,
        total_volume=sum(r.volume for r in results),
        price_change_pct=price_change_pct,
    )

    return GoldRangeResponse(
        start=start_date,
        end=end_date,
        total_records=len(results),
        summary=summary,
        data=results,
    )


# ── Bonus: Model info endpoint ────────────────────────────────────────────────

@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns training metrics for both ML models."""
    return {
        "models": ["GradientBoostingRegressor", "LinearRegression"],
        "features": [
            "date_ordinal", "open", "high", "low", "volume",
            "price_range", "ma_5", "ma_20",
            "lag_1", "lag_3", "lag_5", "volatility_5"
        ],
        "metrics": _state.get("metrics", {}),
    }


# ── Route 3: Statistics ───────────────────────────────────────────────────────

from stats import get_statistics

@app.get("/gold/stats", tags=["Analysis"])
def gold_statistics(
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD) — optional", examples=["2020-01-01"]),
    end:   Optional[str] = Query(None, description="End date (YYYY-MM-DD) — optional",   examples=["2024-12-31"]),
):
    """
    Descriptive statistics for gold price columns (whole dataset or a date range).

    Returns per-column: mean, median, mode, std, variance, min, max,
    Q1/Q2/Q3, IQR, skewness, kurtosis, coefficient of variation.
    Also includes a plain-English interpretation of close price distribution.
    """
    start_date = _parse_date(start) if start else None
    end_date   = _parse_date(end)   if end   else None

    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="'start' must be before or equal to 'end'.")

    result = get_statistics(start_date, end_date)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Route 4: Preprocessing Info ───────────────────────────────────────────────

from preprocess import get_preprocessing_info

@app.get("/gold/preprocess/info", tags=["Analysis"])
def preprocessing_info():
    """
    Full data quality report on the entire gold_prices dataset.

    Returns:
    - Null / missing value counts per column
    - Duplicate record check
    - Outliers via IQR method and Z-score method
    - Sanity checks (high >= low, close within range, no negatives)
    - Zero-volume trading days
    - Overall data quality score and grade (A / B / C / D)
    """
    return get_preprocessing_info()


# ── Route 5: Correlation ──────────────────────────────────────────────────────

from correlation import get_correlation

@app.get("/gold/correlation", tags=["Analysis"])
def gold_correlation(
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD) — optional", examples=["2020-01-01"]),
    end:   Optional[str] = Query(None, description="End date (YYYY-MM-DD) — optional",   examples=["2024-12-31"]),
):
    """
    Pearson and Spearman correlation matrices for all numeric columns.

    Returns:
    - Full Pearson correlation matrix
    - Full Spearman (rank) correlation matrix
    - P-value matrix (statistical significance)
    - All unique column pairs ranked by |r|
    - Close price specific correlations
    - Key insights: strongest pair, weakest pair, significant pairs count
    """
    start_date = _parse_date(start) if start else None
    end_date   = _parse_date(end)   if end   else None

    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="'start' must be before or equal to 'end'.")

    result = get_correlation(start_date, end_date)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result