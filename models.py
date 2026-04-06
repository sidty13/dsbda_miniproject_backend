from pydantic import BaseModel
from typing import Optional, List


class GoldRecord(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    daily_return: Optional[float]
    price_range: float
    predicted_close: Optional[float]


class RangeSummary(BaseModel):
    avg_close: float
    max_close: float
    min_close: float
    avg_daily_return: Optional[float]
    total_volume: int
    price_change_pct: float   


class GoldRangeResponse(BaseModel):
    start: str
    end: str
    total_records: int
    summary: RangeSummary
    data: List[GoldRecord]