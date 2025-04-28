from __future__ import annotations

import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
from typing import Any

from .base_model import BaseModel


class LGBMClassifierWrapper(BaseModel):
    """LightGBM multiclass classifier wrapper mapping classes {-1,0,1}."""

    def __init__(self, booster: lgb.Booster | None = None):
        self.booster = booster

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMClassifierWrapper":
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
        }
        # Map targets {-1,0,1} to {0,1,2} for training
        y_map = y.replace({-1: 0, 0: 1, 1: 2}).astype(int)
        train_data = lgb.Dataset(X, label=y_map)
        self.booster = lgb.train(params, train_data, num_boost_round=300)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not trained")
        probs = self.booster.predict(X)
        # Convert probs to class {-1,0,1}
        classes = np.argmax(probs, axis=1) - 1
        return probs if probs.ndim == 2 else classes

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("No model to save")
        joblib.dump(self.booster, path)

    @classmethod
    def load(cls, path: str) -> "LGBMClassifierWrapper":
        booster = joblib.load(path)
        return cls(booster=booster)

