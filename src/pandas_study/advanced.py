"""Advanced operations — rolling, expanding, ewm, and utilities.

Examples:
    80. .rolling().mean() / .rolling().sum()
    81. .rolling().agg() with multiple functions
    82. .rolling().apply() custom function
    83. .expanding() cumulative window
    84. .ewm() exponential weighted
    85. .duplicated() / .drop_duplicates()
    86. .clip() clamp values
    87. .eval() expression evaluation
    88. pd.Categorical / .astype("category")
    89. .pipe() method chaining
    90. .corr() / .cov() correlation
    91. .idxmax() / .idxmin() index of extrema
    92. pd.get_dummies() one-hot encoding
    93. .where() / .mask() conditional
    94. .explode() list to rows
    95. pd.factorize() / pd.unique()
    96. .align() align two DataFrames
    97. .applymap() / .map() element-wise
    98. pd.option_context() display options
    99. .to_markdown() / .to_string() output
    100. pd.testing utility assertions
"""

import pandas as pd
import numpy as np

from src.pandas_study._interface import DataFrameExample


class AdvancedExamples(DataFrameExample):
    """Rolling windows, expanding, utilities, and output formatting."""

    def run(self) -> dict[str, pd.DataFrame | pd.Series | str | int | float]:
        results: dict = {}
        df = self.make_sample_df()
        s = pd.Series([10, 20, 15, 30, 25, 35, 40, 45])

        # 80. rolling().mean() — moving average
        results["80_rolling_mean"] = pd.DataFrame({
            "value": s,
            "rolling_mean_3": s.rolling(window=3).mean(),
        })

        # 81. rolling().agg() — multiple aggregations
        results["81_rolling_agg"] = s.rolling(window=3).agg(["mean", "std", "min", "max"])

        # 82. rolling().apply() — custom function
        results["82_rolling_apply"] = pd.DataFrame({
            "value": s,
            "range_3": s.rolling(window=3).apply(lambda x: x.max() - x.min()),
        })

        # 83. expanding() — cumulative expanding window
        results["83_expanding"] = pd.DataFrame({
            "value": s,
            "expanding_mean": s.expanding().mean(),
            "expanding_max": s.expanding().max(),
        })

        # 84. ewm() — exponential weighted moving average
        results["84_ewm"] = pd.DataFrame({
            "value": s,
            "ewm_alpha_0.5": s.ewm(alpha=0.5).mean(),
            "ewm_span_3": s.ewm(span=3).mean(),
        })

        # 85. duplicated() / drop_duplicates()
        dup = pd.DataFrame({
            "key": ["a", "b", "a", "c", "b"],
            "val": [1, 2, 1, 3, 2],
        })
        results["85_duplicated"] = dup.duplicated()
        results["85_drop_duplicates"] = dup.drop_duplicates()
        # keep='last' — keep last occurrence
        results["85_drop_duplicates_last"] = dup.drop_duplicates(keep="last")

        # 86. clip() — clamp values to [lower, upper]
        scores = pd.Series([-5, 0, 50, 85, 120])
        results["86_clip"] = pd.DataFrame({
            "original": scores,
            "clipped_0_100": scores.clip(0, 100),
        })

        # 87. eval() — expression evaluation (faster for large DataFrames)
        eval_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        eval_df["sum"] = eval_df.eval("a + b")
        results["87_eval"] = eval_df

        # 88. Categorical dtype — memory-efficient for repeated strings
        cat_series = pd.Series(["low", "medium", "high", "low", "medium"] * 1000)
        results["88_categorical_memory"] = (
            f"object:{cat_series.memory_usage(deep=True)}B "
            f"category:{cat_series.astype('category').memory_usage(deep=True)}B"
        )
        # Ordered categorical
        cat = pd.Categorical(
            ["low", "medium", "high", "low"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
        results["88_ordered_cat"] = pd.DataFrame({"cat": cat, "code": cat.codes})

        # 89. pipe() — method chaining with custom functions
        def add_mean_col(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d["mean_score"] = d["score"].mean()
            return d

        results["89_pipe"] = df.pipe(add_mean_col).head(3)

        # 90. corr() / cov() — correlation and covariance
        corr = df[["age", "score"]].corr()
        results["90_corr"] = corr

        # 91. idxmax() / idxmin() — index label of max/min
        results["91_idxmax"] = pd.Series({
            "score_max_idx": df["score"].idxmax(),
            "score_min_idx": df["score"].idxmin(),
        })

        # 92. pd.get_dummies() — one-hot encoding
        results["92_dummies"] = pd.get_dummies(df[["name", "city"]])
        # drop_first=True avoids dummy variable trap
        results["92_dummies_drop_first"] = pd.get_dummies(df["city"], drop_first=True)

        # 93. where() / mask() — conditional replacement
        results["93_where"] = df["score"].where(df["score"] > 85, other="low")

        # 94. explode() — expand list-like column into rows
        nested = pd.DataFrame({
            "id": [1, 2],
            "tags": [["a", "b"], ["c", "d", "e"]],
        })
        results["94_explode"] = nested.explode("tags")

        # 95. pd.factorize() / pd.unique()
        labels, uniques = pd.factorize(
            pd.Series(["apple", "banana", "apple", "cherry"])
        )
        results["95_factorize"] = pd.DataFrame({
            "original": ["apple", "banana", "apple", "cherry"],
            "label": labels,
        })

        # 96. align() — align two DataFrames by index/columns
        a = pd.DataFrame({"x": [1, 2]}, index=[0, 1])
        b = pd.DataFrame({"y": [3, 4]}, index=[1, 2])
        a_aligned, b_aligned = a.align(b)
        results["96_align_a"] = a_aligned
        results["96_align_b"] = b_aligned

        # 97. map() / applymap() — element-wise mapping
        results["97_map"] = s.map(lambda x: f"val_{x}")
        # For DataFrame element-wise, applymap is deprecated in 2.1+, use map
        results["97_map_df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]}).map(
            lambda x: x * 2
        )

        # 98. option_context() — temporary display options
        with pd.option_context("display.max_rows", 5, "display.max_columns", 3):
            big = pd.DataFrame(np.random.randn(20, 10))
            results["98_display_truncated"] = big.to_string()

        # 99. to_markdown() / to_string() / to_dict()
        mini = df.head(3)
        results["99_to_markdown"] = mini.to_markdown()
        results["99_to_dict"] = str(mini.to_dict(orient="records"))

        # 100. pd.testing — assertion utilities (used in test code)
        a = pd.DataFrame({"x": [1, 2]})
        b = pd.DataFrame({"x": [1, 2]})
        pd.testing.assert_frame_equal(a, b)  # no exception = pass
        results["100_testing_pass"] = "assert_frame_equal passed"

        return results
