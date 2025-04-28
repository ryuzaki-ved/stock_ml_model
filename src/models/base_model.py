from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable
import pandas as pd


class BaseModel(ABC):
    """Abstract base model with a consistent interface."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Iterable[Any]) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        pass

