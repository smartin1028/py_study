"""Tests for pandas_study package — using pytest and mock."""

from unittest.mock import Mock, patch
from io import StringIO

import pandas as pd
import numpy as np
import pytest

from src.pandas_study import (
    IOExamples,
    CoreExamples,
    AggregationExamples,
    CombineExamples,
    CleaningExamples,
    AdvancedExamples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def io_examples():
    return IOExamples()


@pytest.fixture
def core_examples():
    return CoreExamples()


@pytest.fixture
def agg_examples():
    return AggregationExamples()


@pytest.fixture
def combine_examples():
    return CombineExamples()


@pytest.fixture
def cleaning_examples():
    return CleaningExamples()


@pytest.fixture
def advanced_examples():
    return AdvancedExamples()


# ---------------------------------------------------------------------------
# IOExamples
# ---------------------------------------------------------------------------

class TestIOExamples:
    def test_run_returns_all_keys(self, io_examples):
        results = io_examples.run()

        assert "01_read_csv" in results

    def test_read_csv_returns_dataframe(self, io_examples):
        results = io_examples.run()

        assert isinstance(results["01_read_csv"], pd.DataFrame)
        assert list(results["01_read_csv"].columns) == ["name", "age", "city", "score", "joined"]

    def test_read_csv_with_options(self, io_examples):
        results = io_examples.run()

        df = results["02_read_csv_options"]
        assert df.index.name == "name"
        assert list(df.columns) == ["score"]

    def test_read_parquet(self, io_examples):
        results = io_examples.run()

        assert isinstance(results["06_read_parquet"], pd.DataFrame)

    def test_read_json(self, io_examples):
        results = io_examples.run()

        assert isinstance(results["08_read_json"], pd.DataFrame)

    def test_read_sql_returns_filtered(self, io_examples):
        results = io_examples.run()

        df = results["10_read_sql"]
        assert all(df["score"] > 80)

    def test_to_csv_saves_file(self, io_examples, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        assert path.exists()
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_to_excel_creates_file(self, io_examples, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.xlsx"
        df.to_excel(path, index=False)

        assert path.exists()


# ---------------------------------------------------------------------------
# CoreExamples
# ---------------------------------------------------------------------------

class TestCoreExamples:
    def test_run_returns_dataframes(self, core_examples):
        results = core_examples.run()

        assert "11_from_dict" in results
        assert isinstance(results["11_from_dict"], pd.DataFrame)

    def test_from_dict(self, core_examples):
        results = core_examples.run()

        df = results["11_from_dict"]
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]

    def test_from_records(self, core_examples):
        results = core_examples.run()

        df = results["12_from_records"]
        assert len(df) == 2

    def test_series_creation(self, core_examples):
        results = core_examples.run()

        s = results["13_series"]
        assert s.name == "count"
        assert list(s.index) == ["x", "y", "z"]

    def test_describe_statistics(self, core_examples):
        results = core_examples.run()

        desc = results["15_describe"]
        assert "mean" in desc.index
        assert "score" in desc.columns

    def test_head_returns_2_rows(self, core_examples):
        results = core_examples.run()

        assert len(results["16_head"]) == 2

    def test_rename_columns(self, core_examples):
        results = core_examples.run()

        df = results["18_rename"]
        assert "grade" in df.columns

    def test_sort_values_descending(self, core_examples):
        results = core_examples.run()

        df = results["20_sort_by_score"]
        # First score should be >= last score
        assert df["score"].iloc[0] >= df["score"].iloc[-1]

    def test_nlargest_returns_top_n(self, core_examples):
        results = core_examples.run()

        df = results["20_nlargest"]
        assert len(df) == 3

    def test_loc_label_based(self, core_examples):
        results = core_examples.run()

        # loc includes end index
        sliced = results["25_loc_rows"]
        assert isinstance(sliced, pd.DataFrame)

    def test_iloc_position_based(self, core_examples):
        results = core_examples.run()

        sliced = results["26_iloc_rows"]
        # iloc excludes end index, so 0:3 returns 3 rows
        assert len(sliced) == 3

    def test_boolean_filter(self, core_examples):
        results = core_examples.run()

        df = results["28_bool_filter"]
        assert all(df["revenue"] > 150)

    def test_isin_filter(self, core_examples):
        results = core_examples.run()

        df = results["29_isin"]
        assert set(df["product"].unique()) <= {"A", "C"}

    def test_between_filter(self, core_examples):
        results = core_examples.run()

        df = results["30_between"]
        assert all(df["revenue"].between(120, 180))

    def test_query_expression(self, core_examples):
        results = core_examples.run()

        df = results["31_query"]
        assert all(df["revenue"] > 150)
        assert all(df["region"] == "West")

    def test_select_dtypes(self, core_examples):
        results = core_examples.run()

        df_num = results["33_select_number"]
        assert all(df_num.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d)))


# ---------------------------------------------------------------------------
# AggregationExamples
# ---------------------------------------------------------------------------

class TestAggregationExamples:
    def test_groupby_sum(self, agg_examples):
        results = agg_examples.run()

        s = results["34_groupby_sum"]
        assert isinstance(s, pd.Series)
        assert s["Alice"] == 12 + 15 + 18

    def test_groupby_multi_agg(self, agg_examples):
        results = agg_examples.run()

        df = results["35_groupby_multi"]
        assert list(df.columns) == ["sum", "mean", "count"]

    def test_groupby_dict_agg_per_column(self, agg_examples):
        results = agg_examples.run()

        df = results["36_groupby_dict"]
        assert "total_amount" in df.columns
        assert "avg_amount" in df.columns

    def test_transform_keeps_original_shape(self, agg_examples):
        results = agg_examples.run()

        df = results["37_transform"]
        assert "avg_by_cust" in df.columns

    def test_groupby_filter(self, agg_examples):
        results = agg_examples.run()

        df = results["38_filter"]
        assert len(df) > 0

    def test_pivot_table(self, agg_examples):
        results = agg_examples.run()

        pt = results["40_pivot_table"]
        assert "Food" in pt.columns

    def test_crosstab_shape(self, agg_examples):
        results = agg_examples.run()

        ct = results["42_crosstab"]
        assert "Food" in ct.columns

    def test_value_counts(self, agg_examples):
        results = agg_examples.run()

        vc = results["43_value_counts"]
        assert vc.sum() == 8  # total rows in orders

    def test_rank_adds_column(self, agg_examples):
        results = agg_examples.run()

        df = results["45_rank"]
        assert "rank" in df.columns

    def test_cumulative_operations(self, agg_examples):
        results = agg_examples.run()

        cum = results["46_cumulative"]
        assert "cumsum" in cum.columns
        assert "cummax" in cum.columns

    def test_melt_wide_to_long(self, agg_examples):
        results = agg_examples.run()

        melted = results["47_melt"]
        assert "subject" in melted.columns
        assert "score" in melted.columns


# ---------------------------------------------------------------------------
# CombineExamples
# ---------------------------------------------------------------------------

class TestCombineExamples:
    def test_merge_inner(self, combine_examples):
        results = combine_examples.run()

        df = results["49_merge_inner"]
        # Only matching keys: id 2,3,4
        assert set(df["id"]) == {2, 3, 4}

    def test_merge_left_keeps_all_left_keys(self, combine_examples):
        results = combine_examples.run()

        df = results["50_merge_left"]
        assert set(df["id"]) == {1, 2, 3, 4}

    def test_merge_outer_keeps_all_keys(self, combine_examples):
        results = combine_examples.run()

        df = results["50_merge_outer"]
        assert set(df["id"]) == {1, 2, 3, 4, 5}

    def test_concat_rows(self, combine_examples):
        results = combine_examples.run()

        df = results["52_concat_rows"]
        assert len(df) == 6  # left(4 rows) + extra(2 rows)

    def test_join_inner(self, combine_examples):
        results = combine_examples.run()

        df = results["54_join"]
        assert set(df.index) == {2, 3, 4}

    def test_compare_shows_diff(self, combine_examples):
        results = combine_examples.run()

        df = results["56_compare"]
        assert len(df) > 0  # There should be differences

    def test_cut_bins_into_categories(self, combine_examples):
        results = combine_examples.run()

        df = results["58_cut"]
        assert "grade_cut" in df.columns
        assert df["grade_cut"].isna().sum() == 0


# ---------------------------------------------------------------------------
# CleaningExamples
# ---------------------------------------------------------------------------

class TestCleaningExamples:
    def test_isna_detects_missing(self, cleaning_examples):
        results = cleaning_examples.run()

        isna = results["59_isna"]
        assert isna.sum().sum() > 0

    def test_dropna_removes_rows(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["60_dropna_rows"]
        # No NaN should remain in any column
        assert df.isna().sum().sum() == 0

    def test_fillna_with_constant(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["61_fillna_const"]
        assert df["name"].isna().sum() == 0

    def test_ffill_propagates_forward(self, cleaning_examples):
        results = cleaning_examples.run()

        s = results["62_ffill"]
        # NaN should be filled with previous value
        assert s.isna().sum() == 0
        assert s[1] == 1.0

    def test_interpolate_fills_gaps(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["63_interpolate"]
        assert df["interpolated"].isna().sum() == 0

    def test_str_contains_pattern(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["65_str_contains"]
        assert "has_apple" in df.columns

    def test_str_extract_regex(self, cleaning_examples):
        results = cleaning_examples.run()

        extracted = results["67_str_extract"]
        assert extracted.iloc[0, 0] == "apple"

    def test_str_cat_concatenation(self, cleaning_examples):
        results = cleaning_examples.run()

        s = results["69_str_cat"]
        assert s[0] == "A-foo"

    def test_to_datetime_coerces_invalid(self, cleaning_examples):
        results = cleaning_examples.run()

        dt = results["71_to_datetime"]
        assert pd.isna(dt.iloc[2])  # 'invalid' -> NaT

    def test_date_range_creates_sequence(self, cleaning_examples):
        results = cleaning_examples.run()

        dr = results["72_date_range_daily"]
        assert len(dr) == 5
        assert isinstance(dr, pd.DatetimeIndex)

    def test_dt_accessor_extracts_components(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["73_dt_accessor"]
        assert "year" in df.columns
        assert "weekday" in df.columns

    def test_resample_aggregates(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["75_resample"]
        # 12 hours resampled to 4h buckets = 3 rows
        assert len(df) == 3

    def test_shift_diff_create_lags(self, cleaning_examples):
        results = cleaning_examples.run()

        df = results["76_shift_diff"]
        assert pd.isna(df["shift1"].iloc[0])  # First row shift = NaN
        assert pd.isna(df["diff1"].iloc[0])   # First row diff = NaN

    def test_tz_localize_adds_timezone(self, cleaning_examples):
        results = cleaning_examples.run()

        localized = results["78_tz_localize"]
        assert str(localized.tz) == "Asia/Seoul"


# ---------------------------------------------------------------------------
# AdvancedExamples
# ---------------------------------------------------------------------------

class TestAdvancedExamples:
    def test_rolling_mean(self, advanced_examples):
        results = advanced_examples.run()

        df = results["80_rolling_mean"]
        assert "rolling_mean_3" in df.columns
        # First 2 rows are NaN (window=3)
        assert pd.isna(df["rolling_mean_3"].iloc[0])
        assert pd.isna(df["rolling_mean_3"].iloc[1])
        assert not pd.isna(df["rolling_mean_3"].iloc[2])

    def test_rolling_agg_multiple(self, advanced_examples):
        results = advanced_examples.run()

        df = results["81_rolling_agg"]
        assert list(df.columns) == ["mean", "std", "min", "max"]

    def test_expanding_mean(self, advanced_examples):
        results = advanced_examples.run()

        df = results["83_expanding"]
        assert "expanding_mean" in df.columns

    def test_ewm_weighted(self, advanced_examples):
        results = advanced_examples.run()

        df = results["84_ewm"]
        assert "ewm_alpha_0.5" in df.columns

    def test_drop_duplicates_removes_dup(self, advanced_examples):
        results = advanced_examples.run()

        df = results["85_drop_duplicates"]
        assert len(df) == 3  # unique keys: a, b, c

    def test_clip_clamps_values(self, advanced_examples):
        results = advanced_examples.run()

        df = results["86_clip"]
        assert df["clipped_0_100"].min() >= 0
        assert df["clipped_0_100"].max() <= 100

    def test_eval_adds_column(self, advanced_examples):
        results = advanced_examples.run()

        df = results["87_eval"]
        assert "sum" in df.columns
        assert df["sum"].iloc[0] == 5  # 1 + 4

    def test_categorical_saves_memory(self, advanced_examples):
        results = advanced_examples.run()

        msg = results["88_categorical_memory"]
        # category should use less memory than object
        assert "category" in msg

    def test_get_dummies_one_hot(self, advanced_examples):
        results = advanced_examples.run()

        df = results["92_dummies"]
        assert "city_Seoul" in df.columns  # prefix_city + value

    def test_where_replaces_below_threshold(self, advanced_examples):
        results = advanced_examples.run()

        s = results["93_where"]
        assert (s == "low").any()  # Some values replaced

    def test_explode_expands_list_column(self, advanced_examples):
        results = advanced_examples.run()

        df = results["94_explode"]
        # Original 2 rows with [2, 3] elements -> 5 rows
        assert len(df) == 5

    def test_factorize_encodes_categories(self, advanced_examples):
        results = advanced_examples.run()

        df = results["95_factorize"]
        assert "label" in df.columns
        assert df["label"].dtype.kind == "i"  # integer codes

    def test_align_syncs_indexes(self, advanced_examples):
        results = advanced_examples.run()

        a_aligned = results["96_align_a"]
        assert 2 in a_aligned.index  # index expanded to union

    def test_map_elementwise_transform(self, advanced_examples):
        results = advanced_examples.run()

        s = results["97_map"]
        assert s[0] == "val_10"


# ---------------------------------------------------------------------------
# Mock usage demonstrations — file I/O mocking
# ---------------------------------------------------------------------------

class TestMockExamples:
    """Demonstrate mock/stub patterns for pandas code that touches I/O."""

    def test_mock_read_csv(self):
        """Use patch to mock pd.read_csv so no real file is needed."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        def load_data(path: str) -> pd.DataFrame:
            return pd.read_csv(path)

        with patch("pandas.read_csv", return_value=mock_df) as mock_read:
            result = load_data("/fake/path.csv")

        mock_read.assert_called_once_with("/fake/path.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    def test_mock_read_excel(self):
        """Mock pd.read_excel with spec to catch typos."""
        mock_df = pd.DataFrame({"a": [1]})
        mock_read = Mock(spec=pd.read_excel, return_value=mock_df)

        result = mock_read("data.xlsx")
        pd.testing.assert_frame_equal(result, mock_df)

    def test_mock_to_csv_does_not_write_file(self):
        """Mock DataFrame.to_csv to verify call without real disk I/O."""
        df = pd.DataFrame({"x": [1]})
        df.to_csv = Mock()

        def export(data: pd.DataFrame, path: str) -> None:
            data.to_csv(path, index=False)

        export(df, "out.csv")
        df.to_csv.assert_called_once_with("out.csv", index=False)

    def test_mock_to_sql(self):
        """Mock to_sql so no real database is needed."""
        df = pd.DataFrame({"id": [1]})
        df.to_sql = Mock()

        df.to_sql("table", "sqlite:///fake.db", if_exists="replace")
        df.to_sql.assert_called_once()

    def test_stub_dataframe_for_groupby(self):
        """Stub a DataFrame returned from a dependency to test aggregation logic."""
        stub = pd.DataFrame({
            "region": ["East", "East", "West"],
            "sales": [100, 200, 300],
        })

        def aggregate(d: pd.DataFrame) -> pd.Series:
            return d.groupby("region")["sales"].sum()

        result = aggregate(stub)
        assert result["East"] == 300
        assert result["West"] == 300


# ---------------------------------------------------------------------------
# Edge case & boundary tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_operations(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        assert len(df) == 0
        # Operations on empty should not crash
        assert df.isna().sum().sum() == 0
        assert df.dropna().empty

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        assert len(df) == 1
        assert df["a"].mean() == 1.0

    def test_all_nan_column(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        assert s.mean() is np.nan
        assert s.fillna(0).sum() == 0.0

    def test_duplicate_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
        # Duplicate index — operations should still work
        assert df.loc[0].shape == (2, 1)
