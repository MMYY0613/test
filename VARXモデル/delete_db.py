import sqlite3
from pathlib import Path

# --- 設定 ---
DB_PATH = Path("./data/app.db")

def delete_table(table_name: str):
    """指定したテーブルを削除する"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.commit()
    print(f"🗑️ Table [{table_name}] を削除しました。")

def clean_all_tables(exclude_list: list[str] = None):
    """
    システムテーブル以外をすべて削除してDBを真っ新にする。
    残したいテーブル（例: 苦労して作った timeseries_long など）がある場合は
    exclude_list に指定する。
    """
    if exclude_list is None:
        exclude_list = []
        
    with sqlite3.connect(DB_PATH) as con:
        # テーブル一覧を取得
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
        
        for (table_name,) in tables:
            if table_name not in exclude_list:
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"🗑️ Table [{table_name}] を削除しました。")
        con.commit()

if __name__ == "__main__":
    # --- 使い方1: 特定のテーブル（例: 間違えて作った all_q_merged_raw）だけ消す ---
    delete_table("all_q_merged_raw")

    # --- 使い方2: 不要なテーブルを一括削除して整理する ---
    # 今後の「ファイル別テーブル」運用に不要なものをリストアップ
    # to_exclude = [] # 何も残さず全削除してやり直す場合は空リスト
    
    print("DBのクリーンアップを開始します...")
    clean_all_tables(exclude_list=to_exclude)
    print("✨ クリーンアップ完了")
