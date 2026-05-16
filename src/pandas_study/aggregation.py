"""Aggregation & reshaping — groupby, pivot, crosstab, melt.

Examples:
    34. .groupby().agg() single function
    35. .groupby().agg() multiple functions
    36. .groupby().agg() per-column functions
    37. .groupby().transform()
    38. .groupby().filter()
    39. .groupby().apply()
    40. .pivot_table()
    41. .pivot()
    42. pd.crosstab()
    43. .value_counts()
    44. .nunique()
    45. .rank()
    46. .cumsum() / cumulative
    47. .melt()
    48. .stack() / .unstack()
"""

import pandas as pd
import numpy as np

from src.pandas_study._interface import DataFrameExample


class AggregationExamples(DataFrameExample):
    """GroupBy, aggregation, pivot, and reshaping operations."""

    @staticmethod
    def _orders_df() -> pd.DataFrame:
        """Create an orders dataset."""
        return pd.DataFrame({
            "customer": ["Alice", "Bob", "Alice", "Bob", "Alice", "Charlie", "Bob", "Charlie"],
            "category": ["Food",  "Food", "Drink", "Food", "Drink", "Drink", "Drink", "Food"],
            "amount":   [12,      34,     15,      23,     18,      42,      9,       27],
            "date": pd.date_range("2025-03-01", periods=8),
        })

    def run(self) -> dict[str, pd.DataFrame | pd.Series]:
        results: dict = {}
        orders = self._orders_df()

        # 34. groupby + single aggregation
        results["34_groupby_sum"] = orders.groupby("customer")["amount"].sum()

        # 35. groupby + multiple aggregations
        results["35_groupby_multi"] = orders.groupby("customer")["amount"].agg([
            "sum", "mean", "count"
        ])

        # 36. groupby + per-column agg specs (different functions per column)
        results["36_groupby_dict"] = orders.groupby("category").agg(
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            orders=("amount", "count"),
        )

        # 37. groupby + transform — keep original shape, broadcast group stat
        orders["avg_by_cust"] = orders.groupby("customer")["amount"].transform("mean")
        results["37_transform"] = orders[["customer", "amount", "avg_by_cust"]]

        # 38. groupby + filter — keep groups meeting a condition
        results["38_filter"] = orders.groupby("customer").filter(
            lambda g: g["amount"].sum() >= 50
        )

        # 39. groupby + apply
        results["39_apply"] = orders.groupby("category").apply(
            lambda g: g.nlargest(2, "amount"), include_groups=False
        )

        # 40. pivot_table()
        results["40_pivot_table"] = orders.pivot_table(
            values="amount",
            index="customer",
            columns="category",
            aggfunc="sum",
            fill_value=0,
        )

        # 41. pivot() — reshape, no aggregation (needs unique index+columns)
        unique = orders.groupby(["customer", "category"], as_index=False)["amount"].sum()
        results["41_pivot"] = unique.pivot(index="customer", columns="category", values="amount")

        # 42. pd.crosstab() — frequency/count table
        results["42_crosstab"] = pd.crosstab(orders["customer"], orders["category"])

        # 43. .value_counts()
        results["43_value_counts"] = orders["category"].value_counts()
        # Normalized (proportion)
        results["43_value_counts_norm"] = orders["category"].value_counts(normalize=True)

        # 44. .nunique() — count distinct values
        results["44_nunique"] = orders.groupby("category")["customer"].nunique()

        # 45. .rank()
        orders["rank"] = orders["amount"].rank(ascending=False, method="dense")
        results["45_rank"] = orders[["customer", "amount", "rank"]].sort_values("rank")

        # 46. cumulative operations
        s = pd.Series([10, 20, 15, 30, 25])
        cum = pd.DataFrame({
            "value": s,
            "cumsum": s.cumsum(),
            "cummax": s.cummax(),
            "cummin": s.cummin(),
            "pct_change": s.pct_change(),
        })
        results["46_cumulative"] = cum

        # 47. melt() — wide to long
        wide = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "math": [90, 85],
            "eng":  [88, 92],
        })
        results["47_melt"] = wide.melt(
            id_vars=["name"], var_name="subject", value_name="score"
        )

        # 48. stack() / unstack()
        pivot = wide.set_index("name")
        results["48_stack"] = pivot.stack()
        results["48_unstack"] = pivot.stack().unstack()

        return results
