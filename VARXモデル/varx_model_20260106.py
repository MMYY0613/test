from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import platform

# ==========================
# フォント設定（日本語）
# （必要ならここにフォント設定を書く）
# ==========================

# ==========================
# 基本設定
# ==========================
DATA_PATH = ""

TEST_H = 4          # 最後の TEST_H 期を予測
P = 2               # VARX(p) の次数（1 or 2）
RIDGE = 1.0         # リッジ係数

OUTDIR = Path("")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==========================
# 17セクターのリターン列
# ==========================
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

# ==========================
# 17セクターのセンチメント列
# ==========================
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

# ==========================
# セクター日本語ラベル
# ==========================
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

# ==========================
# マクロ変数の日本語ラベル
# ==========================
JP_MACRO_LABELS = {
    "GDP_LOGDIFF": "GDP",
    "UNEMP_RATE": "失業率",
    "CPI": "CPI",
    "JGB_1Y": "国債1年利回り",
    "JGB_3Y": "国債3年利回り",
    "JGB_10Y": "国債10年利回り",
    "NIKKEI_LOGRET": "日経平均株価",
    "TOPIX_LOGRET": "TOPIX",
    "FX_LOGRET": "円ドル相場",
    "TANKAN_BUSI": "短観・業況判断DI",
}

ENDOG_COLS = RET_COLS + SENT_COLS

# ==========================
# データ読み込み＆マクロ加工
# ==========================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# ログ差分など
df["GDP_LOGDIFF"] = np.log(df["GDP"]).diff()
df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
df["TOPIX_LOGRET"] = np.log(df["TOPIX"]).diff()
df["FX_LOGRET"] = np.log(df["USD_JPY"]).diff()

df["TANKAN_BUSI"] = df["TANKAN_ACT"].combine_first(df["TANKAN_FCST"])

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

used_cols = ENDOG_COLS + MACRO_COLS
df_model = df[used_cols].astype(float).replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()

# ==========================
# VARX 用ユーティリティ
# ==========================
def make_design(endog_df: pd.DataFrame,
                exog_df: pd.DataFrame | None,
                p: int = 1):
    """内生 y_t とラグ・外生を並べた設計行列 X と目的変数 Y を作る"""
    Y = endog_df.values  # (T, m)

    X_parts = [np.ones((len(endog_df), 1))]  # const

    for lag in range(1, p + 1):
        X_parts.append(endog_df.shift(lag).values)

    if exog_df is not None and exog_df.shape[1] > 0:
        X_parts.append(exog_df.values)

    X = np.concatenate(X_parts, axis=1)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    Yv = Y[valid]
    Xv = X[valid]
    idx = endog_df.index[valid]
    return Yv, Xv, idx


def fit_varx_ridge(df_train: pd.DataFrame,
                   endog_cols: list[str],
                   exog_cols: list[str],
                   p: int = 1,
                   ridge: float = 1.0):
    """VARX(p) をリッジ回帰で推定する"""
    endog = df_train[endog_cols]
    exog = df_train[exog_cols] if exog_cols else None

    Y, X, idx = make_design(endog, exog, p=p)

    k = X.shape[1]
    Beta = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ Y)

    labels: list[str] = ["const"]
    for lag in range(1, p + 1):
        for c in endog_cols:
            labels.append(f"lag{lag}_{c}")
    for c in exog_cols:
        labels.append(c)

    assert Beta.shape[0] == len(labels)
    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)

    c_vec = coef.loc["const"].values  # (m,)

    A_list: dict[int, np.ndarray] = {}
    for lag in range(1, p + 1):
        rows = [f"lag{lag}_{c}" for c in endog_cols]
        A_l = coef.loc[rows].values.T  # (m, m)
        A_list[lag] = A_l

    B_mat = coef.loc[exog_cols].values.T  # (m, k_exog)

    model: dict[str, object] = {
        "coef": coef,
        "A_list": A_list,
        "B": B_mat,
        "c": c_vec,
        "endog": endog_cols,
        "exog": exog_cols,
        "train_index": idx,
        "p": p,
    }
    if 1 in A_list:
        model["A1"] = A_list[1]
    if 2 in A_list:
        model["A2"] = A_list[2]
    return model


def forecast_varx(model: dict,
                  df_hist: pd.DataFrame,
                  df_future_exog: pd.DataFrame,
                  steps: int):
    """学習済み VARX で逐次予測（一般の p に対応）"""
    endog: list[str] = model["endog"]
    exog: list[str] = model["exog"]
    A_list: dict[int, np.ndarray] = model["A_list"]
    B: np.ndarray = model["B"]
    c_vec: np.ndarray = model["c"]
    p: int = model["p"]

    y_hist = df_hist[endog].copy()
    preds: list[np.ndarray] = []

    for h in range(steps):
        x_now = df_future_exog.iloc[h][exog].values

        y_hat = c_vec.copy()
        for lag in range(1, p + 1):
            y_lag = y_hist.iloc[-lag].values
            y_hat += A_list[lag] @ y_lag
        y_hat += B @ x_now
        preds.append(y_hat)

        y_hist = pd.concat(
            [
                y_hist,
                pd.DataFrame(
                    [y_hat],
                    index=[df_future_exog.index[h]],
                    columns=endog,
                ),
            ],
            axis=0,
        )

    return pd.DataFrame(preds,
                        index=df_future_exog.index[:steps],
                        columns=endog)

# ==========================
# 可視化系
# ==========================
def to_jp_labels(names: list[str]) -> list[str]:
    labels: list[str] = []
    for s in names:
        if s.startswith("RET_"):
            key = s[len("RET_"):]
            base = JP_SECTOR.get(key, key)
            labels.append(f"{base}（リターン）")
        elif s.startswith("SENT_"):
            key = s[len("SENT_"):]
            base = JP_SECTOR.get(key, key)
            labels.append(f"{base}（センチメント）")
        else:
            labels.append(s)
    return labels


def heatmap(A: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(11, 9))

    vmax = np.nanmax(np.abs(A.values)) if A.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = plt.imshow(
        A.values,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap="bwr",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)

    col_labels = to_jp_labels(list(A.columns))
    row_labels = to_jp_labels(list(A.index))

    plt.xticks(range(len(A.columns)), col_labels, rotation=90, ha="center")
    plt.yticks(range(len(A.index)), row_labels)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pred(df_all: pd.DataFrame,
              train_end: pd.Timestamp,
              pred_s: pd.Series,
              col: str,
              outpath: Path):
    """1本分の実績 vs 予測グラフ"""
    actual = df_all[col]

    plt.figure(figsize=(9, 4))
    plt.plot(actual.index, actual.values, marker="o",
             linewidth=1.5, label="実績")

    x_pred = [train_end] + list(pred_s.index)
    y_pred = [actual.loc[train_end]] + list(pred_s.values)
    plt.plot(x_pred, y_pred, marker="o",
             linewidth=2.0, label="予測")

    plt.axvline(train_end, linestyle="--", linewidth=1.0, label="学習終了")

    ax = plt.gca()
    idx = df_all.index
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

    ax.grid(True, which="both", axis="both",
            linestyle="--", linewidth=0.5, alpha=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_train_fit(
    df_train: pd.DataFrame,
    df_fitted: pd.DataFrame,
    col: str,
    outpath: Path,
    r2: float | None = None,
):
    """
    train 期間の実績 vs 当てはめ値を重ねて描画
    df_train : 元スケールの train データ（train_raw）
    df_fitted: 元スケールの当てはめ値（同じ列名）
    col      : 対象列（例: 'RET_FOODS'）
    r2       : その列の R^2（あればタイトルに表示）
    """
    actual = df_train.loc[df_fitted.index, col]
    fitted = df_fitted[col]

    plt.figure(figsize=(9, 4))
    plt.plot(
        actual.index,
        actual.values,
        marker="o",
        linewidth=1.5,
        label="実績（train）",
    )
    plt.plot(
        fitted.index,
        fitted.values,
        marker="o",
        linewidth=2.0,
        label="当てはめ値",
    )

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

    title = f"Train当てはめ：{col}"
    if col.startswith("RET_"):
        sec = JP_SECTOR[col[4:]]
        title = f"Train当てはめ：{sec}（リターン）"
    if r2 is not None:
        title += f"（R²={r2:.2f}）"

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ==========================
# インパルス応答
# ==========================
def impulse_response(A1_mat: np.ndarray,
                     shock_idx: int,
                     steps: int,
                     scale: float = 1.0):
    """VAR(1) 内生ショック IRF"""
    m = A1_mat.shape[0]
    irf = np.zeros((steps + 1, m))
    irf[0, shock_idx] = scale

    for h in range(1, steps + 1):
        irf[h] = A1_mat @ irf[h - 1]
    return irf


def impulse_response_var2(A1: np.ndarray,
                          A2: np.ndarray,
                          shock_idx: int,
                          steps: int,
                          scale: float = 1.0):
    """VAR(2) 内生ショック IRF"""
    m = A1.shape[0]
    irf = np.zeros((steps + 1, m))

    y_minus1 = np.zeros(m)
    y0 = np.zeros(m)
    y0[shock_idx] = scale
    irf[0] = y0

    y_prev2 = y_minus1
    y_prev1 = y0

    for h in range(1, steps + 1):
        y_h = A1 @ y_prev1 + A2 @ y_prev2
        irf[h] = y_h
        y_prev2, y_prev1 = y_prev1, y_h

    return irf


def impulse_response_exog(A1_mat: np.ndarray,
                          B_mat: np.ndarray,
                          exog_idx: int,
                          steps: int,
                          scale: float = 1.0):
    """VAR(1) 外生ショック IRF"""
    m = A1_mat.shape[0]
    d0 = B_mat[:, exog_idx] * scale
    irf = np.zeros((steps + 1, m))
    irf[0, :] = d0

    for h in range(1, steps + 1):
        irf[h, :] = A1_mat @ irf[h - 1, :]
    return irf


def impulse_response_exog_var2(A1: np.ndarray,
                               A2: np.ndarray,
                               B: np.ndarray,
                               exog_idx: int,
                               steps: int,
                               scale: float = 1.0):
    """VAR(2) 外生ショック IRF"""
    m = A1.shape[0]
    irf = np.zeros((steps + 1, m))

    y_minus1 = np.zeros(m)
    y0 = B[:, exog_idx] * scale
    irf[0] = y0

    y_prev2 = y_minus1
    y_prev1 = y0

    for h in range(1, steps + 1):
        y_h = A1 @ y_prev1 + A2 @ y_prev2
        irf[h] = y_h
        y_prev2, y_prev1 = y_prev1, y_h
    return irf


def plot_irf_ret(irf_df: pd.DataFrame,
                 sectors_jp: list[str],
                 title: str,
                 outpath: Path):
    plt.figure(figsize=(8, 4))
    for s in sectors_jp:
        plt.plot(irf_df.index, irf_df[s], marker="o", label=s)

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("期（horizon）")
    plt.ylabel("標準化リターン")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ====================================================
# ブートストラップ IRF 信頼区間（残差ブートストラップ）
# ====================================================
def bootstrap_irf_ci(
    train_std: pd.DataFrame,
    resid_std: np.ndarray,
    model: dict,
    endog_cols: list[str],
    exog_cols: list[str],
    shock_var: str,
    steps: int = 12,
    B: int = 500,
    ridge: float = 1.0,
    random_state: int | None = 42,
):
    """
    残差ブートストラップで IRF の信頼区間を求める。

    train_std : 標準化済みの train データ（endog + exog）
    resid_std: 標準化スケールでの残差 (T_eff, m) = Yv_train - Yhat_train_std
    model    : 元の VARX モデル（fit_varx_ridge の戻り値）
    endog_cols: 内生変数の列名（ENDOG_COLS）
    exog_cols : 外生変数の列名（MACRO_COLS）
    shock_var : ショックを与える内生変数名（例: "SENT_FOODS"）
    steps     : IRF の horizon
    B         : ブートストラップの回数
    ridge     : リッジ係数
    random_state: 乱数シード（再現性用）
    """

    rng = np.random.default_rng(random_state)

    p = model["p"]
    A_list: dict[int, np.ndarray] = model["A_list"]
    B_mat: np.ndarray = model["B"]
    c_vec: np.ndarray = model["c"]

    m = len(endog_cols)

    # train_std から内生・外生を numpy で取り出し
    endog_train = train_std[endog_cols].values  # (T_train, m)
    exog_train = train_std[exog_cols].values    # (T_train, k_exog)
    T_train = endog_train.shape[0]

    # 残差は make_design の有効サンプル数 T_eff = T_train - p 程度
    T_eff = resid_std.shape[0]
    if T_eff != T_train - p:
        print(f"[WARN] T_eff={T_eff}, T_train-p={T_train-p} でズレがあります。")
    shock_idx = endog_cols.index(shock_var)

    # IRF を B 回分ためる配列: (B, steps+1, m)
    irf_samples = np.zeros((B, steps + 1, m), dtype=float)

    for b in range(B):
        # --- 1. 残差を行単位でブートストラップ ---
        idx_boot = rng.integers(0, T_eff, size=T_eff)
        resid_boot = resid_std[idx_boot, :]  # (T_eff, m)

        # --- 2. ブートストラップ標本の内生系列 y_boot を生成 ---
        y_boot = np.zeros_like(endog_train)

        # 最初の p 期は元データそのまま（初期条件固定）
        y_boot[:p, :] = endog_train[:p, :]

        # t = p,...,T_train-1 をループ
        for i in range(T_eff):
            t = p + i  # 対応する時点

            u_star = resid_boot[i, :]  # その期のブートストラップ残差 (m,)

            # 決定論的部分： c + A1 y_{t-1} + ... + Ap y_{t-p} + B x_t
            y_det = c_vec.copy()
            for lag in range(1, p + 1):
                y_det += A_list[lag] @ y_boot[t - lag, :]

            x_t = exog_train[t, :]  # 外生変数（標準化済）
            y_det += B_mat @ x_t

            y_boot[t, :] = y_det + u_star

        # --- 3. 生成した y_boot で VARX を再推定 ---
        df_boot_std = pd.DataFrame(
            np.hstack([y_boot, exog_train]),
            index=train_std.index,
            columns=endog_cols + exog_cols,
        )

        model_b = fit_varx_ridge(
            df_train=df_boot_std,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=p,
            ridge=ridge,
        )

        # --- 4. そのモデルで IRF を計算 ---
        if p == 1:
            A1_b = model_b["A1"]
            irf_b = impulse_response(
                A1_mat=A1_b,
                shock_idx=shock_idx,
                steps=steps,
                scale=1.0,
            )
        elif p == 2:
            A1_b = model_b["A1"]
            A2_b = model_b["A2"]
            irf_b = impulse_response_var2(
                A1=A1_b,
                A2=A2_b,
                shock_idx=shock_idx,
                steps=steps,
                scale=1.0,
            )
        else:
            raise ValueError(f"IRFは p={p} にまだ対応させていない")

        irf_samples[b, :, :] = irf_b  # (steps+1, m)

        if (b + 1) % 50 == 0:
            print(f"[BOOT] {b+1}/{B} 完了")

    # --- 5. 分位点から信頼区間を計算 ---
    irf_lower = np.percentile(irf_samples, 2.5, axis=0)   # (steps+1, m)
    irf_upper = np.percentile(irf_samples, 97.5, axis=0)  # (steps+1, m)

    horizons = np.arange(steps + 1)

    irf_ci_lower = pd.DataFrame(irf_lower, index=horizons, columns=endog_cols)
    irf_ci_upper = pd.DataFrame(irf_upper, index=horizons, columns=endog_cols)

    return irf_ci_lower, irf_ci_upper, irf_samples


def plot_irf_ret_with_ci(
    irf_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_df: pd.DataFrame,
    sectors_jp: list[str],
    title: str,
    outpath: Path,
):
    """
    セクター別 IRF（標準化）＋ ブートストラップ信頼区間を描画
    irf_df  : ベースライン IRF （行: horizon, 列: 日本語セクター名）
    lower_df, upper_df: 同じ形式で、下側・上側の区間
    """
    plt.figure(figsize=(8, 4))
    x = irf_df.index.values

    for s in sectors_jp:
        y = irf_df[s].values
        lo = lower_df[s].values
        hi = upper_df[s].values

        plt.plot(x, y, marker="o", linewidth=1.8, label=s)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("期（horizon）")
    plt.ylabel("標準化リターン")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# ==========================
# メイン処理
# ==========================

# 学習/テスト分割
df_model_raw = df_model
split_idx = max(len(df_model_raw) - TEST_H, 1)
train_raw = df_model_raw.iloc[:split_idx].copy()
test_raw = df_model_raw.iloc[split_idx:].copy()

# 標準化
means = train_raw.mean()
stds = train_raw.std(ddof=0).replace(0, 1.0)

train_std = (train_raw - means) / stds
test_std = (test_raw - means) / stds

# VARX(p) 推定
model = fit_varx_ridge(
    df_train=train_std,
    endog_cols=ENDOG_COLS,
    exog_cols=MACRO_COLS,
    p=P,
    ridge=RIDGE,
)

# 予測（逐次）
pred_std = forecast_varx(
    model=model,
    df_hist=train_std,
    df_future_exog=test_std[MACRO_COLS],
    steps=len(test_std),
)

# 係数を保存
coef = model["coef"]
A1 = pd.DataFrame(model["A1"], index=ENDOG_COLS, columns=ENDOG_COLS)
B = pd.DataFrame(model["B"], index=ENDOG_COLS, columns=MACRO_COLS)
c_vec = pd.Series(model["c"], index=ENDOG_COLS, name="c")

coef.to_csv(OUTDIR / "coef_table.csv", encoding="utf-8-sig")
A1.to_csv(OUTDIR / "A1.csv", encoding="utf-8-sig")
B.to_csv(OUTDIR / "B.csv", encoding="utf-8-sig")
c_vec.to_csv(OUTDIR / "c.csv", header=True, encoding="utf-8-sig")

# 予測を元スケールに戻す
pred = pred_std.copy()
for col in ENDOG_COLS:
    pred[col] = pred_std[col] * stds[col] + means[col]

# RMSE（リターン17本だけ）
rmse_ret = np.sqrt(((pred[RET_COLS] - test_raw[RET_COLS]) ** 2).mean(axis=0))
rmse_ret.to_csv(OUTDIR / "rmse_returns.csv", header=True, encoding="utf-8-sig")

# ==========================
# 相関係数（フルサンプル）
# ==========================
corr_all = df_model.corr()
corr_all.to_csv(OUTDIR / "corr_all.csv", encoding="utf-8-sig")

corr_ret = df_model[RET_COLS].corr()
corr_ret.to_csv(OUTDIR / "corr_returns.csv", encoding="utf-8-sig")
heatmap(
    corr_ret,
    "相関ヒートマップ（17セクター・リターン同士）",
    OUTDIR / "corr_returns_heatmap.png",
)

corr_sent = df_model[SENT_COLS].corr()
corr_sent.to_csv(OUTDIR / "corr_sentiments.csv", encoding="utf-8-sig")
heatmap(
    corr_sent,
    "相関ヒートマップ（17セクター・センチメント同士）",
    OUTDIR / "corr_sentiments_heatmap.png",
)

corr_ret_macro = df_model[RET_COLS + MACRO_COLS].corr()
corr_ret_macro.to_csv(OUTDIR / "corr_returns_macros.csv", encoding="utf-8-sig")
corr_ret_macro_block = corr_ret_macro.loc[RET_COLS, MACRO_COLS]
heatmap(
    corr_ret_macro_block,
    "相関ヒートマップ（17セクター・リターン × マクロ）",
    OUTDIR / "corr_returns_macros_heatmap.png",
)

# ==========================
# 学習サンプルに対する 1期先予測（当てはめ）と R^2
# ==========================
Yv_train, Xv_train, idx_train = make_design(
    endog_df=train_std[ENDOG_COLS],
    exog_df=train_std[MACRO_COLS],
    p=P,
)

Beta = model["coef"].values  # (k, m)

Yhat_train_std = Xv_train @ Beta  # (T_eff, m)

# ブートストラップ用 残差（標準化スケール）
resid_std = Yv_train - Yhat_train_std  # (T_eff, m)

fitted_train_std = pd.DataFrame(
    Yhat_train_std,
    index=idx_train,
    columns=ENDOG_COLS,
)

fitted_train = fitted_train_std.copy()
for col in ENDOG_COLS:
    fitted_train[col] = fitted_train_std[col] * stds[col] + means[col]

r2_dict = {}
for col in RET_COLS:
    y_true = train_raw.loc[idx_train, col]
    y_pred = fitted_train[col]

    sse = ((y_true - y_pred) ** 2).sum()
    sst = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - sse / sst
    r2_dict[col] = r2

r2_ret = pd.Series(r2_dict, name="R2_train")
r2_ret.to_csv(OUTDIR / "r2_train_returns.csv", header=True, encoding="utf-8-sig")

r2_ret_jp = r2_ret.copy()
r2_ret_jp.index = [JP_SECTOR[c.replace("RET_", "")] for c in r2_ret.index]
r2_ret_jp.to_csv(OUTDIR / "r2_train_returns_jp.csv", header=["R2_train"], encoding="utf-8-sig")

print("\n=== Train R^2（リターン：日本語）===")
print(r2_ret_jp.sort_values(ascending=False))

for col in RET_COLS:
    sector_jp = JP_SECTOR[col[4:]]
    r2_val = r2_ret[col]

    plot_train_fit(
        df_train=train_raw,
        df_fitted=fitted_train,
        col=col,
        outpath=OUTDIR / f"train_fit_{sector_jp}.png",
        r2=r2_val,
    )

# ベースライン：RW & 平均
baseline_rw = pd.DataFrame(index=pred.index, columns=RET_COLS, dtype=float)
last_train_y = train_raw[RET_COLS].iloc[-1]
for c in RET_COLS:
    baseline_rw[c] = last_train_y[c]
rmse_rw = np.sqrt(((baseline_rw - test_raw[RET_COLS]) ** 2).mean(axis=0))

baseline_mean = pd.DataFrame(index=pred.index, columns=RET_COLS, dtype=float)
mean_train_y = train_raw[RET_COLS].mean(axis=0)
for c in RET_COLS:
    baseline_mean[c] = mean_train_y[c]
rmse_mean = np.sqrt(((baseline_mean - test_raw[RET_COLS]) ** 2).mean(axis=0))

comparison = pd.DataFrame({
    "VARX_RMSE": rmse_ret,
    "RW_RMSE": rmse_rw,
    "MEAN_RMSE": rmse_mean,
})
comparison["Improvement_vs_RW_%"] = 100 * (
    comparison["RW_RMSE"] - comparison["VARX_RMSE"]
) / comparison["RW_RMSE"]
comparison["Improvement_vs_MEAN_%"] = 100 * (
    comparison["MEAN_RMSE"] - comparison["VARX_RMSE"]
) / comparison["MEAN_RMSE"]

comparison_jp = comparison.copy()
comparison_jp.index = [JP_SECTOR[c.replace("RET_", "")] for c in comparison.index]
comparison_jp = comparison_jp.rename(columns={
    "VARX_RMSE": "VARX_RMSE（本モデル）",
    "RW_RMSE": "RW_RMSE（前期値固定）",
    "MEAN_RMSE": "MEAN_RMSE（平均モデル）",
    "Improvement_vs_RW_%": "RW比改善率（％）",
    "Improvement_vs_MEAN_%": "平均モデル比改善率（％）",
})
comparison_jp.to_csv(OUTDIR / "rmse_comparison_multi_jp.csv", encoding="utf-8-sig")

print("\n=== RMSE比較（リターン：日本語）===")
print(comparison_jp.sort_values("RW比改善率（％）", ascending=False))

pred_vs_act = pd.concat(
    [pred.add_prefix("pred_"), test_raw.add_prefix("act_")],
    axis=1,
)
pred_vs_act.to_csv(OUTDIR / "pred_vs_actual.csv", encoding="utf-8-sig")

meta = {
    "TEST_p": P,
    "TEST_H": TEST_H,
    "RIDGE": RIDGE,
    "ENDOG_COLS": ENDOG_COLS,
    "RET_COLS": RET_COLS,
    "SENT_COLS": SENT_COLS,
    "MACRO_COLS": MACRO_COLS,
    "n_total": int(len(df_model_raw)),
    "n_train": int(len(train_raw)),
    "n_test": int(len(test_raw)),
    "index": [str(x.date()) for x in df_model_raw.index],
}
(OUTDIR / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# --------------------------
# ヒートマップたち
# --------------------------
B_ret = B.loc[RET_COLS, MACRO_COLS].copy()
B_ret.index = [JP_SECTOR[c[4:]] for c in RET_COLS]
B_ret.columns = [JP_MACRO_LABELS[c] for c in MACRO_COLS]
heatmap(
    B_ret,
    "B ヒートマップ（マクロ → 17セクター・リターン）",
    OUTDIR / "B_ret_heatmap.png",
)

B_sent = B.loc[SENT_COLS, MACRO_COLS].copy()
B_sent.index = [JP_SECTOR[c[5:]] for c in SENT_COLS]
B_sent.columns = [JP_MACRO_LABELS[c] for c in MACRO_COLS]
heatmap(
    B_sent,
    "B ヒートマップ（マクロ → 17セクター・センチメント）",
    OUTDIR / "B_sent_heatmap.png",
)

heatmap(
    A1,
    "A1 ヒートマップ（17セクター・リターン＋センチメント）",
    OUTDIR / "A1_heatmap.png",
)

A1_block1 = A1.loc[RET_COLS, SENT_COLS]
heatmap(
    A1_block1,
    "A1 ヒートマップ（17セクター・センチメント → リターン）",
    OUTDIR / "A1_sent_to_ret_heatmap.png",
)

A1_block2 = A1.loc[SENT_COLS, SENT_COLS]
heatmap(
    A1_block2,
    "A1 ヒートマップ（17セクター・センチメント → センチメント）",
    OUTDIR / "A1_sent_to_sent_heatmap.png",
)

A1_block3 = A1.loc[RET_COLS, RET_COLS]
heatmap(
    A1_block3,
    "A1 ヒートマップ（17セクター・リターン → リターン）",
    OUTDIR / "A1_ret_to_ret_heatmap.png",
)

A1_block4 = A1.loc[SENT_COLS, RET_COLS]
heatmap(
    A1_block4,
    "A1 ヒートマップ（17セクター・リターン → センチメント）",
    OUTDIR / "A1_ret_to_sent_heatmap.png",
)

for col in RET_COLS:
    plot_pred(
        df_all=df_model_raw,
        train_end=train_raw.index[-1],
        pred_s=pred[col],
        col=col,
        outpath=OUTDIR / f"pred_{JP_SECTOR[col[4:]]}.png",
    )

print(f"[DONE] VARX({P}) 17セクター")
print("出力先:", OUTDIR.resolve())

# ==========================
# 追加：全サンプルで VARX を推定
# ==========================
ENDOG_ALL = RET_COLS + SENT_COLS
EXOG_ALL = MACRO_COLS

df_full = df[ENDOG_ALL + EXOG_ALL].copy()
df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna()

print("full sample length:", len(df_full))

RIDGE_FULL = 1.0
df_full_std = (df_full - df_full.mean()) / df_full.std(ddof=0)

model_full = fit_varx_ridge(
    df_train=df_full_std,
    endog_cols=ENDOG_ALL,
    exog_cols=EXOG_ALL,
    p=P,
    ridge=RIDGE_FULL,
)

coef_full = model_full["coef"]
A1_full = pd.DataFrame(model_full["A1"], index=ENDOG_ALL, columns=ENDOG_ALL)
B_full = pd.DataFrame(model_full["B"], index=ENDOG_ALL, columns=EXOG_ALL)
c_full = pd.Series(model_full["c"], index=ENDOG_ALL, name="const")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print("\n=== full VARX coef_table（一部） ===")
print(coef_full.round(4).head(20))
print("\n=== A1（ラグ1係数行列） ===")
print(A1_full.round(4))
print("\n=== B（マクロ係数行列） ===")
print(B_full.round(4))
print("\n=== const ベクトル ===")
print(c_full.round(4))

coef_full.to_csv(OUTDIR / "coef_full_table.csv", encoding="utf-8-sig")
A1_full.to_csv(OUTDIR / "A1_full.csv", encoding="utf-8-sig")
B_full.to_csv(OUTDIR / "B_full.csv", encoding="utf-8-sig")
c_full.to_csv(OUTDIR / "c_full.csv", encoding="utf-8-sig")

print("\n[INFO] full-sample VARX coefficients saved in", OUTDIR)
print("RMSE (returns):")
print(rmse_ret.sort_values())

B_full_ret = B_full.loc[RET_COLS, MACRO_COLS].copy()
B_full_ret.index = [JP_SECTOR[c[4:]] for c in RET_COLS]
B_full_ret.columns = [JP_MACRO_LABELS[c] for c in MACRO_COLS]
heatmap(
    B_full_ret,
    "B_full ヒートマップ（マクロ → 17セクター・リターン）",
    OUTDIR / "B_full_ret_heatmap.png",
)

B_full_sent = B_full.loc[SENT_COLS, MACRO_COLS].copy()
B_full_sent.index = [JP_SECTOR[c[5:]] for c in SENT_COLS]
B_full_sent.columns = [JP_MACRO_LABELS[c] for c in MACRO_COLS]
heatmap(
    B_full_sent,
    "B_full ヒートマップ（マクロ → 17セクター・センチメント）",
    OUTDIR / "B_full_sent_heatmap.png",
)

heatmap(
    A1_full,
    "A1_full ヒートマップ（17セクター・リターン＋センチメント）",
    OUTDIR / "A1_full_heatmap.png",
)

A1_full_block1 = A1_full.loc[RET_COLS, SENT_COLS]
heatmap(
    A1_full_block1,
    "A1_full ヒートマップ（17セクター・センチメント → リターン）",
    OUTDIR / "A1_full_sent_to_ret_heatmap.png",
)

A1_full_block2 = A1_full.loc[SENT_COLS, SENT_COLS]
heatmap(
    A1_full_block2,
    "A1_full ヒートマップ（17セクター・センチメント → センチメント）",
    OUTDIR / "A1_full_sent_to_sent_heatmap.png",
)

A1_full_block3 = A1_full.loc[RET_COLS, RET_COLS]
heatmap(
    A1_full_block3,
    "A1_full ヒートマップ（17セクター・リターン → リターン）",
    OUTDIR / "A1_full_ret_to_ret_heatmap.png",
)

A1_full_block4 = A1_full.loc[SENT_COLS, RET_COLS]
heatmap(
    A1_full_block4,
    "A1_full ヒートマップ（17セクター・リターン → センチメント）",
    OUTDIR / "A1_full_ret_to_sent_heatmap.png",
)

# ==========================
# インパルス応答（内生ショック）＋ ブートストラップ CI
# ==========================
IRF_STEPS = 12
shock_var = "SENT_FOODS"
shock_idx = ENDOG_COLS.index(shock_var)

p_irf = model["p"]
if p_irf == 1:
    A1_mat = model["A1"]
    irf = impulse_response(A1_mat, shock_idx, steps=IRF_STEPS, scale=1.0)
elif p_irf == 2:
    A1_mat = model["A1"]
    A2_mat = model["A2"]
    irf = impulse_response_var2(A1_mat, A2_mat, shock_idx, steps=IRF_STEPS, scale=1.0)
else:
    raise ValueError(f"IRFは p={p_irf} にまだ対応させていない")

ret_indices = [ENDOG_COLS.index(c) for c in RET_COLS]
irf_ret = pd.DataFrame(
    irf[:, ret_indices],
    index=range(IRF_STEPS + 1),
    columns=[JP_SECTOR[c.replace("RET_", "")] for c in RET_COLS],
)
irf_ret.index.name = "horizon"
irf_ret.to_csv(OUTDIR / f"irf_{shock_var}_to_returns_std.csv", encoding="utf-8-sig")

print(f"\n=== IRF: {shock_var} ショック → セクターリターン（標準化） ===")
print(irf_ret.head())

sectors_to_plot = ["食品", "銀行", "不動産"]
plot_irf_ret(
    irf_ret,
    sectors_to_plot,
    title=f"{shock_var} に +1σショック → リターンのインパルス応答（標準化）",
    outpath=OUTDIR / f"irf_{shock_var}_to_returns.png",
)

# ---- 残差ブートストラップで IRF の信頼区間 ----
B_BOOT = 500  # 重ければ 200 とかでもOK

irf_ci_lower_all, irf_ci_upper_all, irf_samples = bootstrap_irf_ci(
    train_std=train_std,
    resid_std=resid_std,
    model=model,
    endog_cols=ENDOG_COLS,
    exog_cols=MACRO_COLS,
    shock_var=shock_var,
    steps=IRF_STEPS,
    B=B_BOOT,
    ridge=RIDGE,
    random_state=42,
)

# リターン部分だけ抜き出し、日本語ラベルへ
ret_indices = [ENDOG_COLS.index(c) for c in RET_COLS]
irf_ci_lower_ret = irf_ci_lower_all.iloc[:, ret_indices].copy()
irf_ci_upper_ret = irf_ci_upper_all.iloc[:, ret_indices].copy()
irf_ci_lower_ret.columns = [JP_SECTOR[c.replace("RET_", "")] for c in RET_COLS]
irf_ci_upper_ret.columns = [JP_SECTOR[c.replace("RET_", "")] for c in RET_COLS]

irf_ci_lower_ret.to_csv(
    OUTDIR / f"irf_{shock_var}_to_returns_ci_lower.csv",
    encoding="utf-8-sig",
)
irf_ci_upper_ret.to_csv(
    OUTDIR / f"irf_{shock_var}_to_returns_ci_upper.csv",
    encoding="utf-8-sig",
)

sectors_to_plot = ["食品", "銀行", "不動産"]
plot_irf_ret_with_ci(
    irf_df=irf_ret[sectors_to_plot],
    lower_df=irf_ci_lower_ret[sectors_to_plot],
    upper_df=irf_ci_upper_ret[sectors_to_plot],
    sectors_jp=sectors_to_plot,
    title=f"{shock_var} に +1σショック → リターンのインパルス応答（標準化, 95% CI）",
    outpath=OUTDIR / f"irf_{shock_var}_to_returns_ci.png",
)

print("[DONE] ブートストラップ IRF 信頼区間 計算完了")

# ==========================
# マクロショックのインパルス応答（外生変数）
# ==========================
IRF_STEPS = 12
p_irf = model["p"]
A1_mat = model["A1"]
B_mat = model["B"]
A2_mat = model.get("A2", None)

macro_vars_to_plot = [
    "GDP_LOGDIFF",
    "CPI",
]

for macro_var in macro_vars_to_plot:
    if macro_var not in MACRO_COLS:
        print(f"[WARN] {macro_var} は MACRO_COLS に存在しません。スキップします。")
        continue

    macro_idx = MACRO_COLS.index(macro_var)

    if p_irf == 1:
        irf_macro = impulse_response_exog(
            A1_mat=A1_mat,
            B_mat=B_mat,
            exog_idx=macro_idx,
            steps=IRF_STEPS,
            scale=1.0,
        )
    elif p_irf == 2 and A2_mat is not None:
        irf_macro = impulse_response_exog_var2(
            A1=A1_mat,
            A2=A2_mat,
            B=B_mat,
            exog_idx=macro_idx,
            steps=IRF_STEPS,
            scale=1.0,
        )
    else:
        raise ValueError(f"マクロIRFは p={p_irf} に未対応です")

    ret_indices = [ENDOG_COLS.index(c) for c in RET_COLS]
    irf_macro_ret = pd.DataFrame(
        irf_macro[:, ret_indices],
        index=range(IRF_STEPS + 1),
        columns=[JP_SECTOR[c.replace("RET_", "")] for c in RET_COLS],
    )
    irf_macro_ret.index.name = "horizon"

    irf_macro_ret.to_csv(
        OUTDIR / f"irf_macro_{macro_var}_to_returns_std.csv",
        encoding="utf-8-sig",
    )

    print(f"\n=== IRF: {macro_var} +1σショック → セクターリターン（標準化） ===")
    print(irf_macro_ret.head())

    sectors_to_plot = ["食品", "銀行", "不動産"]
    plot_irf_ret(
        irf_macro_ret,
        sectors_to_plot,
        title=f"{macro_var} に +1σショック → リターンのインパルス応答（標準化）",
        outpath=OUTDIR / f"irf_macro_{macro_var}_to_returns.png",
    )