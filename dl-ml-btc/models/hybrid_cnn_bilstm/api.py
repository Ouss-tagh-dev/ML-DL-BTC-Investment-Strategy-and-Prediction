
#!/usr/bin/env python3
"""
api.py — FastAPI deployment for BTC News Impact Predictor
Install : pip install fastapi uvicorn
Run     : uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from btc_predictor import BTCNewsPredictor, FocalLoss  # noqa: F401

app = FastAPI(title="BTC News Impact Predictor", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = BTCNewsPredictor("./models/hybrid_cnn_bilstm")


class NewsRequest(BaseModel):
    text             : str
    sentiment_score  : float           = Field(default=0.0, ge=-1.0, le=1.0)
    category         : str             = Field(default="OTHER")
    source           : str             = Field(default="Unknown")
    price_change_24h : float           = Field(default=0.0)
    severity         : int             = Field(default=1, ge=1, le=10)
    price            : float           = Field(default=95_000.0)
    # Optional daily context
    volume_news      : Optional[float] = 5.0
    avg_sentiment    : Optional[float] = 0.0
    std_sentiment    : Optional[float] = 0.1
    avg_price_change : Optional[float] = 0.0
    max_severity     : Optional[float] = 5.0
    avg_severity     : Optional[float] = 3.0
    news_momentum    : Optional[float] = 5.0
    sent_momentum    : Optional[float] = 0.0
    sent_volatility  : Optional[float] = 0.0
    volume_accel     : Optional[float] = 0.0


@app.get("/")
def health():
    return {
        "status" : "running",
        "model"  : predictor.cfg.get("model_version"),
        "auc"    : predictor.cfg.get("test_auc"),
        "f1"     : predictor.cfg.get("test_f1_macro"),
    }


@app.post("/predict")
def predict(req: NewsRequest):
    t0     = time.time()
    result = predictor.predict(**req.model_dump())
    return {**result, "latency_ms": round((time.time() - t0) * 1000, 1)}
