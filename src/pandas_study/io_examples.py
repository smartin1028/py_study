"""File I/O examples — read/write CSV, Excel, Parquet, JSON, and more.

Examples:
    01. read_csv()
    02. read_csv() with options (index_col, usecols, dtype)
    03. to_csv()
    04. read_excel()
    05. to_excel()
    06. read_parquet()
    07. to_parquet()
    08. read_json()
    09. to_json()
    10. read_sql() via sqlite3 + read_sql_query()
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from src.pandas_study._interface import DataFrameExample


class IOExamples(DataFrameExample):
    """Read/write data from various file formats."""

    def run(self) -> dict[str, pd.DataFrame | pd.Series | str]:
        results: dict[str, pd.DataFrame | pd.Series | str] = {}
        tmp = Path(tempfile.gettempdir())

        # 01. read_csv()
        csv_path = tmp / "sample.csv"
        self.make_sample_df().to_csv(csv_path, index=False)
        results["01_read_csv"] = pd.read_csv(csv_path)

        # 02. read_csv() with options — index_col, usecols, dtype
        results["02_read_csv_options"] = pd.read_csv(
            csv_path,
            index_col="name",
            usecols=["name", "score"],
            dtype={"score": "float64"},
        )

        # 03. to_csv()
        df = self.make_sample_df()
        out_csv = tmp / "output.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        results["03_to_csv"] = f"saved to {out_csv}"

        # 04. read_excel()
        xlsx_path = tmp / "sample.xlsx"
        df.to_excel(xlsx_path, index=False)
        results["04_read_excel"] = pd.read_excel(xlsx_path)

        # 05. to_excel() with multiple sheets
        out_xlsx = tmp / "multi.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.iloc[:3].to_excel(writer, sheet_name="first3", index=False)
            df.iloc[3:].to_excel(writer, sheet_name="last2", index=False)
        results["05_to_excel_multi_sheet"] = (
            f"sheets={pd.read_excel(out_xlsx, sheet_name=None).keys()}"
        )

        # 06. read_parquet()
        pq_path = tmp / "sample.parquet"
        df.to_parquet(pq_path, index=False)
        results["06_read_parquet"] = pd.read_parquet(pq_path)

        # 07. to_parquet() with compression
        out_pq = tmp / "compressed.parquet"
        df.to_parquet(out_pq, compression="snappy")
        results["07_to_parquet_compressed"] = f"size={out_pq.stat().st_size}B"

        # 08. read_json()
        json_path = tmp / "sample.json"
        df.to_json(json_path, orient="records", force_ascii=False, date_format="iso")
        results["08_read_json"] = pd.read_json(json_path)

        # 09. to_json() with orient options
        results["09_to_json_split"] = df.to_json(orient="split", date_format="iso")
        results["09_to_json_index"] = df.iloc[:2].to_json(orient="index", date_format="iso")

        # 10. read_sql() — using sqlite3 in-memory
        conn = sqlite3.connect(":memory:")
        df.to_sql("users", conn, index=False, if_exists="replace")
        results["10_read_sql"] = pd.read_sql_query(
            "SELECT name, score FROM users WHERE score > 80", conn
        )
        conn.close()

        return results
