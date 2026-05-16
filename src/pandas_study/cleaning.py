"""Data cleaning — missing data, string handling, datetime operations.

Examples:
    59. .isna() / .notna()
    60. .dropna()
    61. .fillna() with constant
    62. .fillna() with method (ffill)
    63. .interpolate()
    64. .replace()
    65. .str.contains()
    66. .str.replace()
    67. .str.extract() regex
    68. .str.split() / .str.get()
    69. .str.cat() concatenation
    70. .str.len() / .str.lower() / .str.strip()
    71. pd.to_datetime()
    72. pd.date_range()
    73. .dt.year / .dt.month / .dt.day / .dt.weekday
    74. .dt.strftime() format
    75. .resample() time-series aggregation
    76. .shift() / .diff()
    77. pd.Timedelta operations
    78. .tz_localize() / .tz_convert()
    79. pd.cut() datetime bins (period)
"""

import pandas as pd
import numpy as np

from src.pandas_study._interface import DataFrameExample


class CleaningExamples(DataFrameExample):
    """Missing data, string, and datetime handling."""

    @staticmethod
    def _missing_df() -> pd.DataFrame:
        """Dataset with missing values."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", np.nan, "Diana"],
            "score": [85.0, np.nan, 78.0, np.nan],
            "city": ["Seoul", np.nan, "Seoul", "Busan"],
        })

    @staticmethod
    def _text_series() -> pd.Series:
        return pd.Series([
            "apple_123", "BANANA_456", "cherry_789", "DATE_000", np.nan,
        ])

    def run(self) -> dict[str, pd.DataFrame | pd.Series]:
        results: dict = {}

        # --- Missing Data ---

        missing = self._missing_df()

        # 59. isna() / notna() — detect missing
        results["59_isna"] = missing.isna()
        results["59_notna"] = missing.notna()

        # 60. dropna() — remove rows/columns with NaN
        results["60_dropna_rows"] = missing.dropna()
        results["60_dropna_cols"] = missing.dropna(axis=1)
        # subset: only check specific columns
        results["60_dropna_subset"] = missing.dropna(subset=["name", "score"])

        # 61. fillna() with constant
        results["61_fillna_const"] = missing.fillna({"name": "unknown", "score": 0})

        # 62. fillna() with ffill / bfill
        s = pd.Series([1, np.nan, np.nan, 4, np.nan])
        results["62_ffill"] = s.ffill()
        results["62_bfill"] = s.bfill()

        # 63. interpolate() — linear interpolation
        s2 = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan])
        results["63_interpolate"] = pd.DataFrame({
            "original": s2,
            "interpolated": s2.interpolate(),
        })

        # 64. replace() — substitute values
        results["64_replace"] = missing.replace({np.nan: "N/A"})

        # --- String Handling ---

        text = self._text_series()

        # 65. str.contains() — pattern match
        results["65_str_contains"] = pd.DataFrame({
            "text": text,
            "has_apple": text.str.contains("apple", na=False),
        })

        # 66. str.replace() — regex substitution
        results["66_str_replace"] = text.str.replace(r"_\d+", "", regex=True)

        # 67. str.extract() — capture groups
        results["67_str_extract"] = text.str.extract(r"^([a-z]+)", flags=0)

        # 68. str.split() + str.get()
        results["68_str_split"] = pd.DataFrame({
            "text": text,
            "prefix": text.str.split("_").str.get(0),
            "suffix": text.str.split("_").str.get(1),
        })

        # 69. str.cat() — concatenate
        first = pd.Series(["A", "B", "C"])
        last = pd.Series(["foo", "bar", "baz"])
        results["69_str_cat"] = first.str.cat(last, sep="-")

        # 70. str.len(), str.lower(), str.strip()
        messy = pd.Series(["  Hello  ", "  World  ", None])
        results["70_str_ops"] = pd.DataFrame({
            "original": messy,
            "len": messy.str.len(),
            "lower": messy.str.lower(),
            "strip": messy.str.strip(),
        })

        # --- Datetime ---

        # 71. pd.to_datetime() — parse strings
        dates = pd.Series(["2025-01-01", "2025-06-15", "invalid"])
        results["71_to_datetime"] = pd.to_datetime(dates, errors="coerce")

        # 72. pd.date_range() — generate date sequences
        results["72_date_range_daily"] = pd.date_range("2025-01-01", periods=5, freq="D")
        # Business month end
        results["72_date_range_bme"] = pd.date_range("2025-01-01", periods=4, freq="BME")

        # 73. dt accessor — extract year, month, day, weekday
        dr = pd.date_range("2025-01-15", periods=5, freq="W")
        results["73_dt_accessor"] = pd.DataFrame({
            "date": dr,
            "year": dr.year,
            "month": dr.month,
            "day": dr.day,
            "weekday": dr.weekday,  # 0=Monday
            "month_name": dr.month_name(),
        })

        # 74. dt.strftime() — format datetime to string
        results["74_strftime"] = dr.strftime("%Y-%m-%d %A")

        # 75. resample() — time-series aggregation
        ts = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=12, freq="h"),
            "value": range(12),
        })
        ts_daily = ts.set_index("date").resample("4h").sum()
        results["75_resample"] = ts_daily

        # 76. shift() / diff()
        s3 = pd.Series([10, 20, 30, 40])
        results["76_shift_diff"] = pd.DataFrame({
            "value": s3,
            "shift1": s3.shift(1),
            "diff1": s3.diff(),
        })

        # 77. Timedelta operations
        t0 = pd.Timestamp("2025-01-01")
        results["77_timedelta"] = pd.DataFrame({
            "date": pd.date_range(t0, periods=4, freq="D"),
            "plus_7d": pd.date_range(t0, periods=4, freq="D") + pd.Timedelta(days=7),
            "diff": pd.Series([
                pd.Timedelta(days=7),
                pd.Timedelta(hours=36),
                pd.Timedelta(minutes=90),
                pd.Timedelta(seconds=3600),
            ]),
        })

        # 78. tz_localize() / tz_convert() — timezone handling
        naive = pd.date_range("2025-01-01", periods=3, freq="D")
        localized = naive.tz_localize("Asia/Seoul")
        results["78_tz_localize"] = localized
        results["78_tz_convert"] = localized.tz_convert("US/Eastern")

        # 79. pd.cut() with datetime — group dates into periods
        period_dates = pd.date_range("2025-01-01", periods=6, freq="ME")
        results["79_cut_datetime"] = pd.DataFrame({
            "date": period_dates,
            "quarter": pd.cut(
                period_dates,
                bins=pd.date_range("2025-01-01", "2025-12-31", freq="QE"),
            ),
        })

        return results
