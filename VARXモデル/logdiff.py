from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========= 基本パス =========
DATA_PATH = "./data/all_q_merged.csv"  # ←必要ならフルパスに変更
OUTDIR = Path("./eda_transformed_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ========= マクロ変数 日本語ラベル =========
MACRO_JP = {
    "GDP": "GDP",
    "UNEMP_RATE": "失業率",
    "CPI": "CPI",
    "JGB_1Y": "国債1年利回り",
    "JGB_3Y": "国債3年利回り",
    "JGB_10Y": "国債10年利回り",
    "NIKKEI": "日経平均株価",
    "TOPIX": "TOPIX",
    "USD_JPY": "円ドル相場"
}

# ========= データ読み込み =========
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# 数値列だけ抽出
num_cols = df.select_dtypes(include=[np.number]).columns

# ========= 各列を log-diff or diff に変換 =========
df_trans = pd.DataFrame(index=df.index)

for col in num_cols:
    s = df[col].astype(float)

    # 正の値だけなら log-diff、それ以外は通常差分
    if (s > 0).all():
        new_name = col + "_logdiff"
        df_trans[new_name] = np.log(s).diff()
    else:
        new_name = col + "_pct_change"
        df_trans[new_name] = s.pct_change()

# diff なので最初の期は NaN → 全列 NaN の行は落とす
df_trans = df_trans.dropna(how="all")

# ★ 変換後データを CSV 保存
trans_csv_path = OUTDIR / "all_q_transformed.csv"
df_trans.to_csv(trans_csv_path, encoding="utf-8-sig")
print("変換後データを保存:", trans_csv_path.resolve())

# ========= 可視化（1列1ファイル） =========
for col in df_trans.columns:
    plt.figure(figsize=(9, 3))
    plt.plot(df_trans.index, df_trans[col], marker="o", linewidth=1.2)

    # ---- タイトルを決める（マクロだけ日本語）----
    title = col
    base_name = col

    # suffix を取って元の列名に戻す
    if col.endswith("_logdiff"):
        base_name = col[:-8]  # "_logdiff" は8文字
        suffix_jp = "(対数差分)"
    elif col.endswith("_diff"):
        base_name = col[:-5]  # "_diff" は5文字
        suffix_jp = "(変化率)"
    else:
        suffix_jp = ""

    if base_name in MACRO_JP:
        # マクロは日本語ラベル
        title = MACRO_JP[base_name] + suffix_jp
    else:
        # それ以外は元の列名
        title = col

    plt.title(title)
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(OUTDIR / f"{col}.png", dpi=150)
    plt.close()

print("プロットを保存しました:", OUTDIR.resolve())