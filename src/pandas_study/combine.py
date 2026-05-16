"""Combine DataFrames — merge, join, concat, compare.

Examples:
    49. pd.merge() inner join
    50. pd.merge() left/right join
    51. pd.merge() on multiple keys
    52. pd.concat() vertical (rows)
    53. pd.concat() horizontal (columns)
    54. .join() index-based
    55. pd.merge_asof() nearest-key join
    56. .compare() diff two DataFrames
    57. .combine_first() fill missing from another
    58. pd.cut() / pd.qcut() binning
"""

import pandas as pd
import numpy as np

from src.pandas_study._interface import DataFrameExample


class CombineExamples(DataFrameExample):
    """Merge, join, concatenate, and compare DataFrames."""

    @staticmethod
    def _left() -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
        })

    @staticmethod
    def _right() -> pd.DataFrame:
        return pd.DataFrame({
            "id": [2, 3, 4, 5],
            "score": [88, 92, 76, 85],
        })

    @staticmethod
    def _extra() -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 6],
            "name": ["Alice", "Frank"],
        })

    def run(self) -> dict[str, pd.DataFrame]:
        results: dict = {}
        left = self._left()
        right = self._right()
        extra = self._extra()

        # 49. merge() inner join — only matching keys
        results["49_merge_inner"] = pd.merge(left, right, on="id", how="inner")

        # 50. merge() left / right / outer join
        results["50_merge_left"] = pd.merge(left, right, on="id", how="left")
        results["50_merge_right"] = pd.merge(left, right, on="id", how="right")
        results["50_merge_outer"] = pd.merge(left, right, on="id", how="outer")

        # 51. merge() on multiple keys
        left_multi = pd.DataFrame({
            "dept": ["HR", "HR", "ENG"],
            "year": [2024, 2025, 2024],
            "budget": [100, 120, 200],
        })
        right_multi = pd.DataFrame({
            "dept": ["HR", "HR", "ENG"],
            "year": [2024, 2025, 2024],
            "headcount": [5, 6, 8],
        })
        results["51_merge_multi_key"] = pd.merge(
            left_multi, right_multi, on=["dept", "year"]
        )

        # 52. concat() vertical — stack rows
        results["52_concat_rows"] = pd.concat([left, extra], ignore_index=True)

        # 53. concat() horizontal — side-by-side columns
        results["53_concat_cols"] = pd.concat(
            [left.set_index("id"), right.set_index("id")], axis=1
        )

        # 54. .join() — index-based join
        l_idx = left.set_index("id")
        r_idx = right.set_index("id")
        results["54_join"] = l_idx.join(r_idx, how="inner")

        # 55. merge_asof() — nearest-key join (useful for time-series)
        trades = pd.DataFrame({
            "time": pd.to_datetime(["09:30:00", "10:00:00", "10:30:00"], format="%H:%M:%S"),
            "price": [100.5, 101.2, 100.8],
        }).set_index("time")
        quotes = pd.DataFrame({
            "time": pd.to_datetime(["09:29:00", "09:59:00", "10:31:00"], format="%H:%M:%S"),
            "bid": [100.0, 101.0, 101.5],
        }).set_index("time")
        results["55_merge_asof"] = pd.merge_asof(
            trades.reset_index().sort_values("time"),
            quotes.reset_index().sort_values("time"),
            on="time",
            direction="backward",
        )

        # 56. .compare() — show differences between two DataFrames
        v1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        v2 = pd.DataFrame({"a": [1, 3], "b": ["x", "z"]})
        results["56_compare"] = v1.compare(v2)

        # 57. .combine_first() — fill NaN from another DataFrame
        main = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, "y", np.nan]})
        fallback = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
        results["57_combine_first"] = main.combine_first(fallback)

        # 58. pd.cut() / pd.qcut() — bin continuous values
        scores = pd.Series([55, 72, 88, 93, 45, 67, 80, 76])
        results["58_cut"] = pd.DataFrame({
            "score": scores,
            "grade_cut": pd.cut(scores, bins=[0, 60, 80, 100], labels=["F", "B", "A"]),
            "grade_qcut": pd.qcut(scores, q=3, labels=["low", "mid", "high"]),
        })

        return results
