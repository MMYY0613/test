from __future__ import annotations
import os
import sqlite3
import pandas as pd
from pathlib import Path
import re

# ==========================================
# --- 設定エリア：ここを書き換えるだけ ---
# ==========================================
SQLITE_PATH = "./data/app.db"

# 読み込みたいファイルをリストで指定（csv, xlsx, parquet対応）
TARGET_FILES = [
    "./data/macro.csv",
    "./data/sector.csv",
    "./data/sentiment.csv",
]
# ==========================================

def get_table_name(file_path: str) -> str:
    """ファイル名から安全なテーブル名を生成する (例: sector_14.csv -> sector_14)"""
    stem = Path(file_path).stem
    # 英数字とアンダースコア以外を置換し、先頭が数字ならアンダースコアを付与
    name = re.sub(r'\W+', '_', stem).lower()
    if name[0].isdigit():
        name = f"t_{name}"
    return name

def load_any_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ Skip: {path.name} (Not Found)")
        return pd.DataFrame()

    ext = path.suffix.lower()
    
    # 1. 拡張子による読み込み
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        print(f"⚠️ Unsupported extension: {ext}")
        return pd.DataFrame()

    # 2. Date列の正規化
    if "Date" not in df.columns:
        for cand in ["date", "DATE", "日付", "time"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    
    if "Date" not in df.columns:
        print(f"❌ Error: 'Date' column not found in {path.name}")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # 3. 縦持ち(Long)変換
    value_cols = [c for c in df.columns if c not in ["Date"]]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    long_df = df.melt(id_vars=["Date"], var_name="series", value_name="value")
    long_df = long_df.dropna(subset=["Date", "value"])
    
    # 日付をSQLite検索用に文字列化
    long_df["Date"] = long_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return long_df

def upsert_to_sqlite(df: pd.DataFrame, table_name: str):
    if df.empty:
        return

    with sqlite3.connect(SQLITE_PATH) as con:
        # テーブルごとにPRIMARY KEYを設定して作成
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT,
                series TEXT,
                value REAL,
                PRIMARY KEY (date, series)
            )
        """)
        
        # UPSERT実行
        data_list = df[["Date", "series", "value"]].values.tolist()
        con.executemany(f"""
            INSERT OR REPLACE INTO {table_name} (date, series, value)
            VALUES (?, ?, ?)
        """, data_list)
        
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
        con.commit()

def main():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    
    for file_path in TARGET_FILES:
        table_name = get_table_name(file_path)
        df_long = load_any_file(file_path)
        
        if not df_long.empty:
            upsert_to_sqlite(df_long, table_name)
            print(f"✅ Loaded: {Path(file_path).name} -> Table: [{table_name}] ({len(df_long)} rows)")

if __name__ == "__main__":
    main()