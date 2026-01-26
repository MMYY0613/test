from __future__ import annotations

import os
import sqlite3
import numpy as np
import pandas as pd

XLSX_PATH = "./data/all_q_merged.xlsx"
TABLE = "all_q_merged_raw"
SQLITE_PATH = "./data/app.db"


def load_excel(xlsx_path: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=0)
    new_cols = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = new_cols

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if "GDP" in df.columns and "GDP_LOGDIFF" not in df.columns:
        df["GDP_LOGDIFF"] = np.log(df["GDP"]).diff()
    if "NIKKEI" in df.columns and "NIKKEI_LOGRET" not in df.columns:
        df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
    if "TOPIX" in df.columns and "TOPIX_LOGRET" not in df.columns:
        df["TOPIX_LOGRET"] = np.log(df["TOPIX"]).diff()
    if "USD_JPY" in df.columns and "FX_LOGRET" not in df.columns:
        df["FX_LOGRET"] = np.log(df["USD_JPY"]).diff()
    if (
        "TANKAN_ACT" in df.columns
        and "TANKAN_FCST" in df.columns
        and "TANKAN_BUSI" not in df.columns
    ):
        df["TANKAN_BUSI"] = df["TANKAN_ACT"].combine_first(df["TANKAN_FCST"])
    return df


def main():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)

    df = load_excel(XLSX_PATH)
    df = add_derived(df)

    # pandasのdatetimeをSQLiteに入れるときは文字列化しておくと安定
    df = df.copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    con = sqlite3.connect(SQLITE_PATH)
    try:
        df.to_sql(TABLE, con, if_exists="replace", index=False)

        # Dateにインデックス（SQLiteは "Date" でも Date でもOKだが統一でダブルクォート）
        con.execute(f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_date ON {TABLE}("Date");')
        con.commit()
    finally:
        con.close()

    print("OK:", TABLE, df.shape, "->", SQLITE_PATH)


if __name__ == "__main__":
    main()
