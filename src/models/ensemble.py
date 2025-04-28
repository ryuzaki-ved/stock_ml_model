from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from .base_model import BaseModel


class ProbabilityAveragingEnsemble(BaseModel):
    """Simple ensemble that averages class probabilities across base models."""

    def __init__(self, models: List[BaseModel]):
        self.models = models

    def fit(self, X: pd.DataFrame, y) -> "ProbabilityAveragingEnsemble":
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = [m.predict(X) for m in self.models]
        avg = np.mean(probs, axis=0)
        return avg

    def save(self, path: str) -> None:
        # Saving composite models would typically need a directory and metadata; omitted for brevity
        raise NotImplementedError("Composite save not implemented")

    @classmethod
    def load(cls, path: str) -> "ProbabilityAveragingEnsemble":
        raise NotImplementedError("Composite load not implemented")

