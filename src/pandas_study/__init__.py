"""pandas_study — pandas 100 examples for study."""

from src.pandas_study._interface import DataFrameExample, SupportsSave
from src.pandas_study.io_examples import IOExamples
from src.pandas_study.core import CoreExamples
from src.pandas_study.aggregation import AggregationExamples
from src.pandas_study.combine import CombineExamples
from src.pandas_study.cleaning import CleaningExamples
from src.pandas_study.advanced import AdvancedExamples

__all__ = [
    "DataFrameExample",
    "SupportsSave",
    "IOExamples",
    "CoreExamples",
    "AggregationExamples",
    "CombineExamples",
    "CleaningExamples",
    "AdvancedExamples",
]
