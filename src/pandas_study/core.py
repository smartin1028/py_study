"""Core DataFrame operations — creation, selection, filtering, sorting.

Examples:
    11. pd.DataFrame() from dict
    12. pd.DataFrame() from list of dicts
    13. pd.Series() creation
    14. .info() — column summary
    15. .describe() — statistical summary
    16. .head() / .tail()
    17. .shape, .columns, .dtypes
    18. .rename() columns
    19. .set_index() / .reset_index()
    20. .sort_values()
    21. .sample()
    22. .copy() vs view
    23. .drop() columns/rows
    24. .astype() type conversion
    25. .loc[] label-based selection
    26. .iloc[] position-based selection
    27. .at[] / .iat[] scalar access
    28. Boolean filtering
    29. .isin() membership
    30. .between() range filter
    31. .query() string expression
    32. .filter() column name
    33. .select_dtypes()
"""

import pandas as pd
import numpy as np

from src.pandas_study._interface import DataFrameExample


class CoreExamples(DataFrameExample):
    """DataFrame creation, selection, filtering, and basic manipulation."""

    @staticmethod
    def _sales_df() -> pd.DataFrame:
        """Create a sales dataset for examples."""
        return pd.DataFrame({
            "product": ["A", "B", "C", "A", "B", "A", "C", "B"],
            "region":  ["East", "West", "East", "West", "East", "West", "East", "West"],
            "revenue": [100, 200, 150, 130, 210, 110, 170, 190],
            "quantity": [10, 20, 15, 13, 21, 11, 17, 19],
            "date": pd.date_range("2025-01-01", periods=8, freq="W"),
        })

    def run(self) -> dict[str, pd.DataFrame | pd.Series | float | int | str | tuple]:
        results: dict = {}
        df = self.make_sample_df()

        # 11. DataFrame from dict
        results["11_from_dict"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # 12. DataFrame from list of dicts
        results["12_from_records"] = pd.DataFrame([
            {"name": "foo", "val": 10},
            {"name": "bar", "val": 20},
        ])

        # 13. Series from list
        results["13_series"] = pd.Series([10, 20, 30], name="count", index=["x", "y", "z"])

        # 14. .info() — returns string summary
        import io
        buf = io.StringIO()
        df.info(buf=buf)
        results["14_info"] = buf.getvalue()

        # 15. .describe()
        results["15_describe"] = df.describe()

        # 16. .head() / .tail()
        results["16_head"] = df.head(2)
        results["16_tail"] = df.tail(2)

        # 17. shape, columns, dtypes
        results["17_shape"] = str(df.shape)
        results["17_columns"] = str(df.columns.tolist())
        results["17_dtypes"] = str(df.dtypes.to_dict())

        # 18. .rename()
        results["18_rename"] = df.rename(columns={"score": "grade", "city": "location"}).head(2)

        # 19. .set_index() / .reset_index()
        indexed = df.set_index("name")
        results["19_set_index"] = indexed
        results["19_reset_index"] = indexed.reset_index().head(3)

        # 20. .sort_values()
        results["20_sort_by_score"] = df.sort_values("score", ascending=False)
        results["20_sort_multi"] = df.sort_values(["city", "age"])
        # 20b. .nlargest() / .nsmallest()
        results["20_nlargest"] = df.nlargest(3, "score")

        # 21. .sample()
        results["21_sample"] = df.sample(3, random_state=42)

        # 22. .copy() — explicit copy for safe modification
        copied = df.copy()
        copied.loc[:, "bonus"] = copied["score"] * 0.1
        results["22_copy"] = copied.head(3)

        # 23. .drop()
        results["23_drop_col"] = df.drop(columns=["joined"])
        results["23_drop_row"] = df.drop(index=[0, 2])

        # 24. .astype()
        results["24_astype"] = df.astype({"age": "float32"}).dtypes.to_frame("dtype")

        # --- Selection ---
        sales = self._sales_df()

        # 25. .loc[] — label-based
        results["25_loc_rows"] = sales.loc[1:3]  # inclusive end
        results["25_loc_cell"] = sales.loc[0, "revenue"]

        # 26. .iloc[] — position-based
        results["26_iloc_rows"] = sales.iloc[0:3]  # exclusive end
        results["26_iloc_cell"] = sales.iloc[0, 2]

        # 27. .at[] / .iat[] — fast scalar access
        results["27_at"] = sales.at[0, "product"]
        results["27_iat"] = sales.iat[0, 0]

        # 28. Boolean filtering
        high_rev = sales[sales["revenue"] > 150]
        results["28_bool_filter"] = high_rev
        # Multiple conditions — use & | and parentheses
        results["28_multi_cond"] = sales[(sales["revenue"] > 150) & (sales["region"] == "East")]

        # 29. .isin()
        results["29_isin"] = sales[sales["product"].isin(["A", "C"])]

        # 30. .between()
        results["30_between"] = sales[sales["revenue"].between(120, 180)]

        # 31. .query() — SQL-like string expression
        results["31_query"] = sales.query("revenue > 150 and region == 'West'")

        # 32. .filter() — select columns by name
        results["32_filter_cols"] = sales.filter(like="revenue")
        results["32_filter_regex"] = sales.filter(regex="^(product|revenue)$")

        # 33. .select_dtypes()
        mixed = pd.DataFrame({
            "int_col": [1, 2],
            "float_col": [1.0, 2.0],
            "str_col": ["a", "b"],
        })
        results["33_select_number"] = mixed.select_dtypes(include="number")
        results["33_select_object"] = mixed.select_dtypes(include=["object", "string"])

        return results
