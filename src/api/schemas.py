"""
Pydantic schemas for API
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    symbols: List[str] = Field(..., example=["RELIANCE", "TCS", "INFY"])
    date: Optional[str] = None

class PredictionResponse(BaseModel):
    symbol: str
    prediction: int  # -1, 0, 1
    confidence: float
    signal: str  # BUY, SELL, HOLD
    timestamp: datetime

class PerformanceMetrics(BaseModel):
    accuracy: float
    sharpe_ratio: float
    win_rate: float
    total_predictions: int
    period_start: datetime
    period_end: datetime