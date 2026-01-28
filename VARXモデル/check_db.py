import sqlite3
import pandas as pd

SQLITE_PATH = "./data/app.db"

def check_db():
    with sqlite3.connect(SQLITE_PATH) as con:
        # 1. システムテーブル以外のテーブル一覧を取得
        tables_df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", 
            con
        )
        tables = tables_df['name'].tolist()
        
        if not tables:
            print("テーブルが見つかりません。ロードスクリプトを先に実行してください。")
            return

        print(f"✅ 発見されたテーブル: {', '.join(tables)}")

        # 2. 各テーブルの中身を深掘り
        for table in tables:
            print(f"\n{'='*50}")
            print(f"📊 TABLE: {table}")
            print(f"{'='*50}")
            
            # 各テーブル内のシリーズごとの件数・期間を集計
            query = f"""
            SELECT 
                series, 
                COUNT(*) as count, 
                MIN(date) as start_date, 
                MAX(date) as end_date 
            FROM {table} 
            GROUP BY series
            """
            try:
                summary = pd.read_sql_query(query, con)
                if summary.empty:
                    print("  (データが空です)")
                else:
                    print(summary.to_string(index=False))
            except Exception as e:
                print(f"  ❌ エラー: {e}")

if __name__ == "__main__":
    check_db()
