import sqlite3
import pandas as pd
import os

DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_prices.db")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GoldUSD.csv")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the gold_prices table and load data from CSV if table is empty."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_prices (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT UNIQUE NOT NULL,   -- stored as YYYY-MM-DD
            open          REAL NOT NULL,
            high          REAL NOT NULL,
            low           REAL NOT NULL,
            close         REAL NOT NULL,
            volume        INTEGER NOT NULL,
            daily_return  REAL,                   -- (close - prev_close) / prev_close * 100
            price_range   REAL,                   -- high - low
            created_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM gold_prices")
    if cursor.fetchone()[0] == 0:
        _load_csv(conn)

    conn.close()


def _load_csv(conn: sqlite3.Connection):
    """Parse the Kaggle CSV and insert all rows into gold_prices."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Parse DD-MM-YY → YYYY-MM-DD
    df["date"] = pd.to_datetime(df["Date"], format="%d-%m-%y").dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    # Feature engineering
    df["daily_return"] = (
        (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    ).round(6)

    df["price_range"] = (df["High"] - df["Low"]).round(6)

    rows = [
        (
            row["date"],
            round(row["Open"], 6),
            round(row["High"], 6),
            round(row["Low"], 6),
            round(row["Close"], 6),
            int(row["Volume"]),
            row["daily_return"] if pd.notna(row["daily_return"]) else None,
            row["price_range"],
        )
        for _, row in df.iterrows()
    ]

    conn.executemany(
        """
        INSERT OR IGNORE INTO gold_prices
            (date, open, high, low, close, volume, daily_return, price_range)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    print(f"[DB] Loaded {len(rows)} rows from CSV into gold_prices table.")