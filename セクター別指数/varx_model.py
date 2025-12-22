# varx_17.py
# 17セクターRET+SENT を内生、マクロ9本を外生にした VARX(1)

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.dates as mdates

# =========================
# フォント設定（日本語）
# =========================
font_path = "/Fonts/YuGothR.ttc"
if Path(font_path).exists():
    fm.fontManager.addfont(font_path)
    jp_font = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = jp_font.get_name()
plt.rcParams["axes.unicode_minus"] = False  # マイナスを正しく表示

# =========================
# 基本設定
# =========================
DATA_PATH = "data/all_q_merged.csv"
OUTDIR = Path("out_varx_17_simple")
TEST_H = 2        # 最後の 2 期を予測
P = 1             # VARX(1)
RIDGE = 1e-1      # リッジ係数（必要なら調整）

OUTDIR.mkdir(parents=True, exist_ok=True)

# 17セクターのリターン列
RET_COLS = [
    "RET_FOODS",
    "RET_ENERGY_RESOURCES",
    "RET_CONSTRUCTION_MATERIALS",
    "RET_RAW_MAT_CHEM",
    "RET_PHARMACEUTICAL",
    "RET_AUTOMOBILES_TRANSP_EQUIP",
    "RET_STEEL_NONFERROUS",
    "RET_MACHINERY",
    "RET_ELEC_APPLIANCES_PRECISION",
    "RET_IT_SERV_OTHERS",
    "RET_ELECTRIC_POWER_GAS",
    "RET_TRANSPORT_LOGISTICS",
    "RET_COMMERCIAL_WHOLESALE",
    "RET_RETAIL_TRADE",
    "RET_BANKS",
    "RET_FIN_EX_BANKS",
    "RET_REAL_ESTATE",
]

# 17セクターのセンチメント列
SENT_COLS = [
    "SENT_FOODS",
    "SENT_ENERGY_RESOURCES",
    "SENT_CONSTRUCTION_MATERIALS",
    "SENT_RAW_MAT_CHEM",
    "SENT_PHARMACEUTICAL",
    "SENT_AUTOMOBILES_TRANSP_EQUIP",
    "SENT_STEEL_NONFERROUS",
    "SENT_MACHINERY",
    "SENT_ELEC_APPLIANCES_PRECISION",
    "SENT_IT_SERV_OTHERS",
    "SENT_ELECTRIC_POWER_GAS",
    "SENT_TRANSPORT_LOGISTICS",
    "SENT_COMMERCIAL_WHOLESALE",
    "SENT_RETAIL_TRADE",
    "SENT_BANKS",
    "SENT_FIN_EX_BANKS",
    "SENT_REAL_ESTATE",
]

# この2つを全部まとめて内生変数に
ENDOG_COLS = RET_COLS + SENT_COLS

# =========================
# データ読み込み＆マクロ加工
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# ---- マクロを log diff に変換する列 ----
df["GDP_LOGDIFF"]   = np.log(df["GDP"]).diff()
df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
df["TOPIX_LOGRET"]  = np.log(df["TOPIX"]).diff()
df["FX_LOGRET"]     = np.log(df["USD_JPY"]).diff()

# 外生に使うマクロ列
MACRO_COLS = [
    "GDP_LOGDIFF",
    "UNEMP_RATE",
    "CPI",
    "JGB_1Y",
    "JGB_3Y",
    "JGB_10Y",
    "NIKKEI_LOGRET",
    "TOPIX_LOGRET",
    "FX_LOGRET",
]

# 型を揃えて NaN/Inf を処理
used_cols = ENDOG_COLS + MACRO_COLS
df_model = df[used_cols].astype(float).replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()  # シンプルに全部揃っている四半期だけ使う

# =========================
# VARX 用のユーティリティ
# =========================
def make_design(endog_df: pd.DataFrame,
                exog_df: pd.DataFrame,
                p: int = 1):
    """定数＋内生ラグ＋外生を並べた設計行列 X と目的変数 Y を作る"""
    Y = endog_df.values                   # (T, m)
    X_parts = [np.ones((len(endog_df), 1))]  # const

    # ここでは p=1 固定
    for lag in range(1, p + 1):
        X_parts.append(endog_df.shift(lag).values)

    if exog_df is not None and exog_df.shape[1] > 0:
        X_parts.append(exog_df.values)

    X = np.concatenate(X_parts, axis=1)  # (T, k)

    # 欠損を含む行は全部落とす
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    Yv = Y[valid]
    Xv = X[valid]
    idx = endog_df.index[valid]
    return Yv, Xv, idx


def fit_varx_ridge(df_train: pd.DataFrame,
                   endog_cols: list[str],
                   exog_cols: list[str],
                   ridge: float = 1.0):
    """VARX(1) をリッジ回帰で推定する"""

    Y, X, idx = make_design(
        df_train[endog_cols],
        df_train[exog_cols],
        p=P
    )

    k = X.shape[1]
    # Beta: (k, m)
    Beta = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ Y)

    labels = (["const"]
              + [f"lag1_{c}" for c in endog_cols]  # p=1 前提
              + exog_cols)
    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)

    # 分解（Y_t = c + A1 Y_{t-1} + B X_t）
    c = coef.loc["const"].values                                # (m,)
    A1 = coef.loc[[f"lag1_{c}" for c in endog_cols]].values.T   # (m,m)
    B = coef.loc[exog_cols].values.T                            # (m, r)

    model = {
        "coef": coef,
        "c": c,
        "A1": A1,
        "B": B,
        "endog": endog_cols,
        "exog": exog_cols,
        "train_index": idx,
    }
    return model


def forecast_varx(model,
                  df_hist: pd.DataFrame,
                  df_future_exog: pd.DataFrame,
                  steps: int) -> pd.DataFrame:
    """学習済み VARX で steps 期先まで逐次予測"""

    endog = model["endog"]
    exog = model["exog"]
    A1 = model["A1"]
    B = model["B"]
    c = model["c"]

    Y_hist = df_hist[endog].copy()
    preds = []

    for h in range(steps):
        y_prev = Y_hist.iloc[-1].values  # 直近の内生ベクトル
        x_now = df_future_exog.iloc[h][exog].values  # 今期の外生
        y_hat = c + A1 @ y_prev + B @ x_now
        preds.append(y_hat)

        # 予測値を履歴に足す（次ステップのラグとして使う）
        Y_hist = pd.concat(
            [
                Y_hist,
                pd.DataFrame([y_hat],
                             index=[df_future_exog.index[h]],
                             columns=endog),
            ],
            axis=0,
        )

    return pd.DataFrame(preds,
                        index=df_future_exog.index[:steps],
                        columns=endog)


# =========================
# 可視化系
# =========================
def heatmap(A: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(11, 9))
    vmax = np.nanmax(np.abs(A.values)) if A.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = plt.imshow(A.values, aspect="auto",
                    cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(A.columns)), A.columns, rotation=60, ha="right")
    plt.yticks(range(len(A.index)), A.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_pred(df_all: pd.DataFrame,
              train_end: pd.Timestamp,
              pred_s: pd.Series,
              col: str,
              outpath: Path):
    """1本分の実績 vs 予測グラフ"""
    actual = df_all[col]

    plt.figure(figsize=(9, 4))
    plt.plot(actual.index, actual.values,
             marker="o", linewidth=1.5, label="実績")
    plt.plot(pred_s.index, pred_s.values,
             marker="o", linewidth=2.0, label="予測")

    plt.axvline(train_end, linestyle="--", linewidth=1.0, label="学習終了")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")

    plt.legend()
    plt.title(f"{col}: 実績 vs 予測")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# =========================
# メイン処理
# =========================
# 学習/テスト分割
split_idx = max(len(df_model) - TEST_H, 1)
train = df_model.iloc[:split_idx]
test = df_model.iloc[split_idx:]

# VARX(1) 推定
model = fit_varx_ridge(
    df_train=train,
    endog_cols=ENDOG_COLS,
    exog_cols=MACRO_COLS,
    ridge=RIDGE,
)

# 予測
pred = forecast_varx(
    model=model,
    df_hist=train,
    df_future_exog=test[MACRO_COLS],
    steps=len(test),
)

# 係数をそれぞれ保存
coef = model["coef"]
A1 = pd.DataFrame(model["A1"], index=ENDOG_COLS, columns=ENDOG_COLS)
B = pd.DataFrame(model["B"], index=ENDOG_COLS, columns=MACRO_COLS)
c = pd.Series(model["c"], index=ENDOG_COLS, name="c")

coef.to_csv(OUTDIR / "coef_table.csv", encoding="utf-8-sig")
A1.to_csv(OUTDIR / "A1.csv", encoding="utf-8-sig")
B.to_csv(OUTDIR / "B.csv", encoding="utf-8-sig")
c.to_csv(OUTDIR / "c.csv", header=True, encoding="utf-8-sig")

# RMSE（リターン17本だけ）
rmse_ret = np.sqrt(((pred[RET_COLS] - test[RET_COLS]) ** 2).mean(axis=0))
rmse_ret.sort_values().to_csv(
    OUTDIR / "rmse_returns.csv", header=True, encoding="utf-8-sig"
)

# 予測 vs 実績の全体表
pred_vs_act = pd.concat(
    [pred.add_prefix("pred_"), test.add_prefix("act_")],
    axis=1,
)
pred_vs_act.to_csv(OUTDIR / "pred_vs_actual.csv", encoding="utf-8-sig")

# メタ情報
meta = {
    "P": P,
    "TEST_H": TEST_H,
    "RIDGE": RIDGE,
    "ENDOG_COLS": ENDOG_COLS,
    "RET_COLS": RET_COLS,
    "SENT_COLS": SENT_COLS,
    "MACRO_COLS": MACRO_COLS,
    "n_total": int(len(df_model)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "index": [str(x.date()) for x in df_model.index],
}
(OUTDIR / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# A1 ヒートマップ
heatmap(A1, "A1 ヒートマップ（17セクターRET+SENT）",
        OUTDIR / "A1_heatmap.png")

# 各リターンについて予測グラフ
for col in RET_COLS:
    plot_pred(
        df_all=df_model,
        train_end=train.index[-1],
        pred_s=pred[col],
        col=col,
        outpath=OUTDIR / f"pred_{col}.png",
    )

print("[DONE] VARX(1) 17セクター")
print(" 出力先:", OUTDIR.resolve())

# =========================================================
# 追加：全サンプルで VARX を推定して係数行列を保存・表示
# =========================================================

ENDOG_ALL = RET_COLS + SENT_COLS   # 17リターン + 17センチ
EXOG_ALL  = MACRO_COLS             # マクロ

# 内生 + 外生だけに絞ってナン処理
df_full = df[ENDOG_ALL + EXOG_ALL].copy()
df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna()

print("full sample length:", len(df_full))

# リッジ係数（普通の線形回帰に近づけたければ 0.0 か 1e-4 など）
RIDGE_FULL = 1.0

model_full = fit_varx_ridge(
    df_train=df_full,
    endog_cols=ENDOG_ALL,
    exog_cols=EXOG_ALL,
    ridge=RIDGE_FULL,
)

# 係数を DataFrame 化
coef_full = model_full["coef"]              # 1つの大きな係数テーブル
A1_full   = pd.DataFrame(
    model_full["A1"],
    index=ENDOG_ALL,
    columns=ENDOG_ALL,
)
B_full    = pd.DataFrame(
    model_full["B"],
    index=ENDOG_ALL,
    columns=EXOG_ALL,
)
c_full    = pd.Series(
    model_full["c"],
    index=ENDOG_ALL,
    name="const",
)

# ---- コンソールに表示（見やすいように丸め） ----
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print("\n=== full VARX coef_table (一部) ===")
print(coef_full.round(4).head(20))   # 上から数行だけ表示

print("\n=== A1 (ラグ1係数行列) ===")
print(A1_full.round(4))

print("\n=== B (マクロ係数行列) ===")
print(B_full.round(4))

print("\n=== const ベクトル ===")
print(c_full.round(4))

# ---- CSV に保存 ----
coef_full.to_csv(OUTDIR / "coef_full_table.csv", encoding="utf-8-sig")
A1_full.to_csv(OUTDIR / "A1_full.csv", encoding="utf-8-sig")
B_full.to_csv(OUTDIR / "B_full.csv", encoding="utf-8-sig")
c_full.to_csv(OUTDIR / "c_full.csv", encoding="utf-8-sig")

print("\n[INFO] full-sample VARX coefficients saved in", OUTDIR)
print(" RMSE (returns):")
print(rmse_ret.sort_values())