from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor

# ======================================
# 基本設定
# ======================================
DATA_PATH = "”

TEST_H = 4          # 最後の TEST_H 期を予測
RF_N_EST = 500      # ランダムフォレストの木の本数
RF_MAX_DEPTH = None # 深さ制限（Noneなら制限なし）

OUTDIR = Path("")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ======================================
# セクター列（RET）
# ======================================
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
    "RET_TEST",
]

# 日本語ラベル（グラフやCSV用）
JP_SECTOR = {
    "FOODS": "食品",
    "ENERGY_RESOURCES": "エネルギー資源",
    "CONSTRUCTION_MATERIALS": "建設・資材",
    "RAW_MAT_CHEM": "素材・化学",
    "PHARMACEUTICAL": "医薬品",
    "AUTOMOBILES_TRANSP_EQUIP": "自動車・輸送機",
    "STEEL_NONFERROUS": "鉄鋼・非鉄金属",
    "MACHINERY": "機械",
    "ELEC_APPLIANCES_PRECISION": "電機・精密",
    "IT_SERV_OTHERS": "情報通信・サービス",
    "ELECTRIC_POWER_GAS": "電力・ガス",
    "TRANSPORT_LOGISTICS": "運輸・倉庫",
    "COMMERCIAL_WHOLESALE": "商社・卸売",
    "RETAIL_TRADE": "小売",
    "BANKS": "銀行",
    "FIN_EX_BANKS": "金融（除く銀行）",
    "REAL_ESTATE": "不動産",
    "TEST": "テスト",
}

# ======================================
# マクロ列
# ======================================
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
    "TANKAN_BUSI",
]


def load_and_prepare() -> pd.DataFrame:
    """VARXと同じ前処理で df_model を作る"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # ログ差分など（VARXコードと揃える）
    df["GDP_LOGDIFF"] = np.log(df["GDP"]).diff()
    df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
    df["TOPIX_LOGRET"] = np.log(df["TOPIX"]).diff()
    df["FX_LOGRET"] = np.log(df["USD_JPY"]).diff()
    df["TANKAN_BUSI"] = df["TANKAN_ACT"].combine_first(df["TANKAN_FCST"])

    used_cols = RET_COLS + MACRO_COLS
    df_model = df[used_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    return df_model


def make_lagged_xy(
    df_model: pd.DataFrame,
    target: str,
    lag: int = 1,
) -> pd.DataFrame:
    """
    目的変数 target について、1期ラグのリターン + マクロを特徴量にするデータセットを作る。
    y_t = f( y_{t-1}, macro_t )

    戻り値 df_xy は
      [lag_ret, MACRO..., y]
    を列にもつ DataFrame。
    """
    df_feat = pd.DataFrame(index=df_model.index)

    # ラグ付きリターン
    df_feat["lag_ret"] = df_model[target].shift(lag)

    # マクロ
    for c in MACRO_COLS:
        df_feat[c] = df_model[c]

    df_xy = df_feat.copy()
    df_xy["y"] = df_model[target]

    # ラグのせいで最初の数期が NaN なので落とす
    df_xy = df_xy.dropna()
    return df_xy


def plot_pred_with_rw(
    df_full: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    y_pred: pd.Series,
    y_rw: pd.Series,
    outpath: Path,
):
    """
    実績×RF予測×RW（前期値固定）を重ねて可視化
    """
    actual = df_full[target]

    plt.figure(figsize=(9, 4))

    # 実績
    plt.plot(
        actual.index,
        actual.values,
        marker="o",
        linewidth=1.5,
        label="実績",
    )

    # RF予測（train_end から線でつなぐ）
    x_rf = [train_end] + list(y_pred.index)
    y_rf = [actual.loc[train_end]] + list(y_pred.values)
    plt.plot(
        x_rf,
        y_rf,
        marker="o",
        linewidth=2.0,
        label="RF予測",
    )

    # RW予測（train_end から水平線）
    x_rw = [train_end] + list(y_rw.index)
    y_rw_plot = [actual.loc[train_end]] + list(y_rw.values)
    plt.plot(
        x_rw,
        y_rw_plot,
        linestyle="--",
        linewidth=1.5,
        label="RW（前期値固定）",
    )

    # 学習終了の縦線
    plt.axvline(train_end, linestyle="--", linewidth=1.0, label="学習終了")

    ax = plt.gca()
    idx = actual.index
    step = max(len(idx) // 5, 1)
    xticks = list(idx[::step])
    if idx[0] not in xticks:
        xticks.insert(0, idx[0])
    if idx[-1] not in xticks:
        xticks.append(idx[-1])

    ax.set_xticks(xticks)
    ax.set_xlim(idx[0], idx[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=90)

    ax.grid(
        True,
        which="both",
        axis="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )

    sec_jp = JP_SECTOR[target.replace("RET_", "")]
    plt.title(f"RF予測 vs RW vs 実績：{sec_jp}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    df_model = load_and_prepare()
    print("df_model length:", len(df_model))

    # 予測値をためる
    all_pred_rf = pd.DataFrame(index=df_model.index, columns=RET_COLS, dtype=float)
    all_pred_rw = pd.DataFrame(index=df_model.index, columns=RET_COLS, dtype=float)

    rmse_rf_dict: dict[str, float] = {}
    rmse_rw_dict: dict[str, float] = {}

    for target in RET_COLS:
        print(f"\n[INFO] RandomForest for {target}")

        df_xy = make_lagged_xy(df_model, target=target, lag=1)

        # train/test 分割（df_xy 基準で最後 TEST_H 期をテストに）
        split_idx = len(df_xy) - TEST_H
        train_xy = df_xy.iloc[:split_idx]
        test_xy = df_xy.iloc[split_idx:]

        X_train = train_xy.drop(columns=["y"])
        y_train = train_xy["y"]
        X_test = test_xy.drop(columns=["y"]).copy()
        y_test = test_xy["y"].copy()

        # モデル学習
        rf = RandomForestRegressor(
            n_estimators=RF_N_EST,
            max_depth=RF_MAX_DEPTH,
            random_state=0,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # 逐次予測
        pred_vals: list[float] = []

        for i, idx in enumerate(X_test.index):
            x_row = X_test.loc[idx].copy()

            if i > 0:
                # 2期目以降は lag_ret を前期予測値で上書き
                x_row["lag_ret"] = pred_vals[i - 1]

            y_hat = rf.predict(x_row.to_frame().T)[0]
            pred_vals.append(y_hat)

        y_pred = pd.Series(pred_vals, index=X_test.index, name="y_pred")

        # RW（前期値固定）ベースライン
        last_train_y = y_train.iloc[-1]
        y_rw = pd.Series(last_train_y, index=X_test.index, name="y_rw")

        # RMSE
        rmse_rf = float(np.sqrt(((y_pred - y_test) ** 2).mean()))
        rmse_rw = float(np.sqrt(((y_rw - y_test) ** 2).mean()))
        rmse_rf_dict[target] = rmse_rf
        rmse_rw_dict[target] = rmse_rw

        print(f"  RF_RMSE: {rmse_rf:.4f}  /  RW_RMSE: {rmse_rw:.4f}")

        # 元の df_model の index に載せる（最後 TEST_H 期だけ埋まる）
        all_pred_rf.loc[y_pred.index, target] = y_pred.values
        all_pred_rw.loc[y_rw.index, target] = y_rw.values

        # 予測グラフ（実績×RF×RW）
        train_end = train_xy.index[-1]
        sec_jp = JP_SECTOR[target.replace("RET_", "")]
        out_png = OUTDIR / f"rf_pred_{sec_jp}.png"
        plot_pred_with_rw(
            df_full=df_model,
            target=target,
            train_end=train_end,
            y_pred=y_pred,
            y_rw=y_rw,
            outpath=out_png,
        )

    # --- RMSE表を作成 ---
    rmse_rf_ser = pd.Series(rmse_rf_dict, name="RF_RMSE")
    rmse_rw_ser = pd.Series(rmse_rw_dict, name="RW_RMSE")

    comparison = pd.DataFrame({
        "RF_RMSE": rmse_rf_ser,
        "RW_RMSE": rmse_rw_ser,
    })
    comparison["Improvement_vs_RW_%"] = 100.0 * (
        comparison["RW_RMSE"] - comparison["RF_RMSE"]
    ) / comparison["RW_RMSE"]

    # 英語キー版
    comparison.to_csv(
        OUTDIR / "rmse_rf_vs_rw.csv",
        encoding="utf-8-sig",
    )

    # 日本語ラベル版
    comparison_jp = comparison.copy()
    comparison_jp.index = [JP_SECTOR[c.replace("RET_", "")] for c in comparison_jp.index]
    comparison_jp = comparison_jp.rename(columns={
        "RF_RMSE": "RF_RMSE（ランダムフォレスト）",
        "RW_RMSE": "RW_RMSE（前期値固定）",
        "Improvement_vs_RW_%": "RW比改善率（％）",
    })
    comparison_jp.to_csv(
        OUTDIR / "rmse_rf_vs_rw_jp.csv",
        encoding="utf-8-sig",
    )

    # 予測値を保存
    all_pred_rf.to_csv(OUTDIR / "rf_pred_all_rf.csv", encoding="utf-8-sig")
    all_pred_rw.to_csv(OUTDIR / "rf_pred_all_rw.csv", encoding="utf-8-sig")

    print("\n=== RF vs RW RMSE（日本語） ===")
    print(comparison_jp.sort_values("RW比改善率（％）", ascending=False))

    print("\n[INFO] RF結果 & グラフ saved in", OUTDIR.resolve())


if __name__ == "__main__":
    main()