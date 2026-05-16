"""Public interfaces for pandas_study package."""

import abc
from typing import Protocol

import pandas as pd


class DataFrameExample(abc.ABC):
    """Base class for all example groups."""

    @abc.abstractmethod
    def run(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Execute all examples and return {name: result} mapping."""
        ...

    @staticmethod
    def make_sample_df() -> pd.DataFrame:
        """Create a reusable sample DataFrame for examples."""
        return pd.DataFrame({
            "name": ["alice", "bob", "charlie", "diana", "eve"],
            "age": [25, 30, 35, 28, 22],
            "city": ["Seoul", "Busan", "Seoul", "Incheon", "Busan"],
            "score": [85.5, 92.0, 78.3, 88.7, 95.1],
            "joined": pd.date_range("2024-01-15", periods=5, freq="ME"),
        })


class SupportsSave(Protocol):
    """Protocol for objects that can save to file."""

    def to_csv(self, path: str) -> None: ...
