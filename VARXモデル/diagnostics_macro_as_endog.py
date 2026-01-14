from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================
# 基本設定
# ==========================
# ★ここを自分のパスに変更
DATA_PATH = "./data/all_q_merged.csv"

TEST_H = 4          # 最後の TEST_H 期をテストにする
RIDGE_DEFAULT = 1.0 # デフォルトのリッジ係数

OUTDIR = Path("./diagnostics_reverse_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==========================
# セクター・マクロ列
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

# 日本語ラベル（マクロ）
JP_MACRO = {
    "GDP_LOGDIFF": "GDP（対数差分）",
    "UNEMP_RATE": "失業率",
    "CPI": "消費者物価指数",
    "JGB_1Y": "国債1年利回り",
    "JGB_3Y": "国債3年利回り",
    "JGB_10Y": "国債10年利回り",
    "NIKKEI_LOGRET": "日経平均（対数差分）",
    "TOPIX_LOGRET": "TOPIX（対数差分）",
    "FX_LOGRET": "ドル円レート（対数差分）",
    "TANKAN_BUSI": "日銀短観業況DI",
}

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
# モデルの組み合わせ定義
#   ★マクロが内生(endog)，セクターが外生(exog)
# ==========================
MODEL_SPECS = [
    # ① 最小構成：GDP_LOGDIFF を内生，FOODS を外生
    {
        "name": "gdp__foods_only",
        "p": 1,
        "ridge": 1.0,
        "endog_cols": ["GDP_LOGDIFF"],
        "exog_cols": ["RET_FOODS"],
    },
    # ② GDP_LOGDIFF, CPI を内生，FOODS + BANKS を外生
    {
        "name": "gdp_cpi__foods_banks",
        "p": 1,
        "ridge": 1.0,
        "endog_cols": ["GDP_LOGDIFF", "CPI"],
        "exog_cols": ["RET_FOODS", "RET_BANKS"],
    },
    # ③ 3マクロ (GDP, CPI, 失業率) を内生，3セクターを外生
    {
        "name": "3macros__3sectors",
        "p": 2,
        "ridge": 1.0,
        "endog_cols": ["GDP_LOGDIFF", "CPI", "UNEMP_RATE"],
        "exog_cols": ["RET_FOODS", "RET_BANKS", "RET_REAL_ESTATE"],
    },
    # ④ 3マクロ (GDP, CPI, 失業率) を内生，2セクターを外生
    {
        "name": "3macros__foods_banks",
        "p": 2,
        "ridge": 1.0,
        "endog_cols": ["GDP_LOGDIFF", "CPI", "UNEMP_RATE"],
        "exog_cols": ["RET_FOODS", "RET_BANKS"],
    },
    # 必要に応じて追加
]


# ==========================
# データ読み込み & 前処理
# ==========================
def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # ログ差分など（既存VARXコードと揃える前提）
    df["GDP_LOGDIFF"] = np.log(df["GDP"]).diff()
    df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
    df["TOPIX_LOGRET"] = np.log(df["TOPIX"]).diff()
    df["FX_LOGRET"] = np.log(df["USD_JPY"]).diff()
    df["TANKAN_BUSI"] = df["TANKAN_ACT"].combine_first(df["TANKAN_FCST"])

    used_cols = RET_COLS + MACRO_COLS
    df_model = df[used_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()
    return df_model


# ==========================
# VARX ユーティリティ
# ==========================
def make_design(endog_df: pd.DataFrame,
                exog_df: pd.DataFrame | None,
                p: int = 1):
    """
    VARX用の設計行列 X, 目的変数 Y を作る
      Y_t = c + A1 Y_{t-1} + ... + Ap Y_{t-p} + B X_t + u_t
    """
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


def fit_varx_ridge(df_train_std: pd.DataFrame,
                   endog_cols: list[str],
                   exog_cols: list[str],
                   p: int = 1,
                   ridge: float = 1.0):
    """
    VARX(p) をリッジ回帰で推定
    df_train_std : 標準化済みのデータ（endog + exog を含む）
    """
    endog = df_train_std[endog_cols]
    exog = df_train_std[exog_cols] if exog_cols else None

    Y, X, idx = make_design(endog, exog, p=p)

    k = X.shape[1]
    Beta = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ Y)  # (k, m)

    # 行ラベル
    labels: list[str] = ["c"]   # 定数項 c
    for lag in range(1, p + 1):
        for c in endog_cols:
            labels.append(f"A{lag}_{c}")   # A1_, A2_, ...
    for c in exog_cols:
        labels.append(f"B_{c}")           # B_外生

    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)

    # c, A_l, B を取り出す
    c_vec = coef.loc["c"].values  # (m,)

    A_list: dict[int, np.ndarray] = {}
    for lag in range(1, p + 1):
        rows = [f"A{lag}_{c}" for c in endog_cols]
        A_l = coef.loc[rows].values.T  # (m, m)
        A_list[lag] = A_l

    if exog_cols:
        rows_B = [f"B_{c}" for c in exog_cols]
        B_mat = coef.loc[rows_B].values.T  # (m, len(exog_cols))
    else:
        B_mat = np.zeros((len(endog_cols), 0))

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
                  df_hist_std: pd.DataFrame,
                  df_future_exog_std: pd.DataFrame,
                  steps: int):
    """
    学習済み VARX で逐次予測（標準化スケール）
    """
    endog: list[str] = model["endog"]
    exog: list[str] = model["exog"]
    A_list: dict[int, np.ndarray] = model["A_list"]
    B: np.ndarray = model["B"]
    c_vec: np.ndarray = model["c"]
    p: int = model["p"]

    y_hist = df_hist_std[endog].copy()
    preds: list[np.ndarray] = []

    for h in range(steps):
        x_now = df_future_exog_std.iloc[h][exog].values if exog else np.zeros(B.shape[1])

        y_hat = c_vec.copy()
        for lag in range(1, p + 1):
            y_lag = y_hist.iloc[-lag].values
            y_hat += A_list[lag] @ y_lag
        if exog:
            y_hat += B @ x_now
        preds.append(y_hat)

        y_hist = pd.concat(
            [
                y_hist,
                pd.DataFrame(
                    [y_hat],
                    index=[df_future_exog_std.index[h]],
                    columns=endog,
                ),
            ],
            axis=0,
        )

    return pd.DataFrame(preds,
                        index=df_future_exog_std.index[:steps],
                        columns=endog)


# ==========================
# 係数のブートストラップ感度（行ブートストラップ）
#   ★今回は呼び出さず（ケースBSはスキップ）
# ==========================
def coef_sensitivity_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    ridge: float,
    base_coef: pd.DataFrame,
    B: int = 200,
    random_state: int | None = 0,
):
    """
    ※ 今回は main から呼ばない。
    使うとき用に A/B/c のインデックスに対応させてある。
    """
    rng = np.random.default_rng(random_state)
    T_eff, k = X.shape
    m = Y.shape[1]

    Beta_base = base_coef.values  # (k, m)
    Beta_boot = np.zeros((B, k, m))

    for b in range(B):
        idx = rng.integers(0, T_eff, size=T_eff)
        Xb = X[idx, :]
        Yb = Y[idx, :]

        Beta_b = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(k), Xb.T @ Yb)
        Beta_boot[b, :, :] = Beta_b

    # 成分ごとの標準偏差
    Beta_sd = Beta_boot.std(axis=0)  # (k, m)
    Beta_sd_df = pd.DataFrame(Beta_sd, index=base_coef.index, columns=base_coef.columns)

    coef_sd_mean = float(Beta_sd.mean())
    coef_sd_max = float(Beta_sd.max())

    # ブロックごとの平均SD
    idx_const = (base_coef.index == "c")
    idx_lag   = base_coef.index.str.startswith("A")
    idx_exog  = base_coef.index.str.startswith("B")

    if idx_const.any():
        sd_const_mean = float(Beta_sd[idx_const, :].mean())
    else:
        sd_const_mean = np.nan

    if idx_lag.any():
        sd_lag_mean = float(Beta_sd[idx_lag, :].mean())
    else:
        sd_lag_mean = np.nan

    if idx_exog.any():
        sd_exog_mean = float(Beta_sd[idx_exog, :].mean())
    else:
        sd_exog_mean = np.nan

    # フロベニウス距離（全係数）
    diff = Beta_boot - Beta_base[None, :, :]  # (B, k, m)
    frob = np.linalg.norm(diff, ord="fro", axis=(1, 2))  # (B,)
    frob_mean = float(frob.mean())
    frob_max = float(frob.max())

    return (
        Beta_sd_df,
        coef_sd_mean,
        coef_sd_max,
        sd_const_mean,
        sd_lag_mean,
        sd_exog_mean,
        frob_mean,
        frob_max,
    )


# ==========================
# 係数の感度（平滑化ブートストラップ：ランダムノイズ）
# ==========================
def coef_sensitivity_noise(
    df_train_std: pd.DataFrame,
    endog_cols: list[str],
    exog_cols: list[str],
    p: int,
    ridge: float,
    base_coef: pd.DataFrame,
    noise_scale: float = 0.2,
    B: int = 200,
    random_state: int | None = 0,
):
    """
    平滑化ブートストラップ版の係数感度:
      - 標準化済みの train データにガウスノイズを加える
      - そのたびに VARX を再推定
      - 係数の標準偏差（行列）とブロックごとのSD,
        フロベニウス距離を返す
    """
    rng = np.random.default_rng(random_state)

    target_cols = endog_cols + exog_cols
    Beta_list: list[np.ndarray] = []

    for b in range(B):
        df_pert = df_train_std.copy()
        noise = rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=df_pert[target_cols].shape,
        )
        df_pert[target_cols] = df_train_std[target_cols].values + noise

        model_b = fit_varx_ridge(
            df_train_std=df_pert,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=p,
            ridge=ridge,
        )
        Beta_b = model_b["coef"].values  # (k, m)
        Beta_list.append(Beta_b)

    Beta_arr = np.stack(Beta_list, axis=0)  # (B, k, m)
    Beta_sd = Beta_arr.std(axis=0)          # (k, m)

    Beta_sd_df = pd.DataFrame(
        Beta_sd,
        index=base_coef.index,
        columns=base_coef.columns,
    )

    coef_sd_mean = float(Beta_sd.mean())
    coef_sd_max = float(Beta_sd.max())

    # ブロックごとの平均SD
    idx_const = (base_coef.index == "c")
    idx_lag   = base_coef.index.str.startswith("A")
    idx_exog  = base_coef.index.str.startswith("B")

    if idx_const.any():
        sd_const_mean = float(Beta_sd[idx_const, :].mean())
    else:
        sd_const_mean = np.nan

    if idx_lag.any():
        sd_lag_mean = float(Beta_sd[idx_lag, :].mean())
    else:
        sd_lag_mean = np.nan

    if idx_exog.any():
        sd_exog_mean = float(Beta_sd[idx_exog, :].mean())
    else:
        sd_exog_mean = np.nan

    # フロベニウス距離（全係数）
    Beta_base = base_coef.values
    diff = Beta_arr - Beta_base[None, :, :]
    frob = np.linalg.norm(diff, ord="fro", axis=(1, 2))
    frob_mean = float(frob.mean())
    frob_max = float(frob.max())

    return (
        Beta_sd_df,
        coef_sd_mean,
        coef_sd_max,
        sd_const_mean,
        sd_lag_mean,
        sd_exog_mean,
        frob_mean,
        frob_max,
    )


# ==========================
# VAR のコンパニオン行列 & 固有値
# ==========================
def build_companion(A_list: dict[int, np.ndarray], p: int) -> np.ndarray:
    """
    VAR(p) のコンパニオン行列を作る（外生は無視）。
    """
    m = A_list[1].shape[0]
    comp = np.zeros((m * p, m * p))

    # 1 行目ブロック [A1 A2 ... Ap]
    for lag in range(1, p + 1):
        comp[0:m, (lag - 1) * m: lag * m] = A_list[lag]

    # 下の部分は単位行列をずらして配置
    if p > 1:
        comp[m:, 0:(p - 1) * m] = np.eye((p - 1) * m)

    return comp


def max_abs_eig_of_companion(A_list: dict[int, np.ndarray], p: int) -> float:
    comp = build_companion(A_list, p)
    eigvals = np.linalg.eigvals(comp)
    return float(np.max(np.abs(eigvals)))


# ==========================
# AIC っぽいスコア
# ==========================
def compute_aic_like(resid: np.ndarray, n_params: int) -> float:
    """
    残差（T_eff, m）とパラメータ数から AIC っぽい値を計算。
    厳密ではないが「相対比較用」として使う。
    """
    T_eff, m = resid.shape
    Sigma = (resid.T @ resid) / T_eff  # (m, m)

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return np.nan

    # T * log det(Σ) + 2 k m という形の「AIC風」
    aic_like = T_eff * logdet + 2.0 * n_params * m
    return float(aic_like)

# ==========================
# 描画
# ==========================
def plot_fit_and_forecast(
    df_all: pd.DataFrame,
    train_end: pd.Timestamp,
    fitted_train: pd.DataFrame,
    pred_test: pd.DataFrame,
    col: str,
    outpath: Path,
    jp_name: str | None = None,
):
    """
    実績（全期間）＋ 当てはめ値（train期間）＋ 予測値（test期間）
    を1枚のグラフに描画する。
    - df_all      : 元データ（train+test 全期間）
    - train_end   : 学習終了日（縦線を引く位置）
    - fitted_train: train当てはめ値（endog列のみ）
    - pred_test   : test予測値（endog列のみ）
    - col         : 対象列名（例: 'GDP_LOGDIFF'）
    - jp_name     : 日本語ラベル（なければ col をそのままタイトルに使う）
    """
    actual = df_all[col]

    plt.figure(figsize=(9, 4))

    # 実績（全期間）
    plt.plot(
        actual.index,
        actual.values,
        linewidth=1.5,
        label="実績",
    )

    # 当てはめ値（train）
    plt.plot(
        fitted_train.index,
        fitted_train[col].values,
        linestyle="--",
        linewidth=2.0,
        label="当てはめ（train）",
    )

    # 予測値（test）
    plt.plot(
        pred_test.index,
        pred_test[col].values,
        marker="o",
        linewidth=2.0,
        label="予測（test）",
    )

    # train と test の境目
    plt.axvline(train_end, linestyle=":", linewidth=1.0, label="学習終了")

    ax = plt.gca()
    idx = actual.index
    step = max(len(idx) // 6, 1)
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

    # タイトル（日本語名があればそれを使う）
    name = jp_name if jp_name is not None else col
    plt.title(f"{name}：実績・当てはめ・予測")

    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# ==========================
# メイン処理
# ==========================
def main():
    df_model = load_and_prepare()
    print("df_model length:", len(df_model))

    # train/test 分割（最後 TEST_H 期をテスト）
    split_idx = max(len(df_model) - TEST_H, 1)
    train_raw = df_model.iloc[:split_idx].copy()
    test_raw = df_model.iloc[split_idx:].copy()

    # 標準化は train の平均・分散で
    means = train_raw.mean()
    stds = train_raw.std(ddof=0).replace(0, 1.0)

    train_std = (train_raw - means) / stds
    test_std = (test_raw - means) / stds

    summary_rows = []

    for spec in MODEL_SPECS:
        name = spec["name"]
        p = spec["p"]
        ridge = spec.get("ridge", RIDGE_DEFAULT)
        endog_cols = spec["endog_cols"]
        exog_cols = spec["exog_cols"]

        print(f"\n===== モデル: {name} (p={p}, ridge={ridge}) =====")
        print("  内生変数(endog):", endog_cols)
        print("  外生変数(exog):", exog_cols)

        # --- VARX 推定 ---
        model = fit_varx_ridge(
            df_train_std=train_std[endog_cols + exog_cols],
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=p,
            ridge=ridge,
        )

        coef: pd.DataFrame = model["coef"]
        A_list: dict[int, np.ndarray] = model["A_list"]

        # --- train 当てはめ & 残差（標準化） ---
        endog_train_std = train_std[endog_cols]
        exog_train_std = train_std[exog_cols] if exog_cols else None

        Yv_train, Xv_train, idx_train = make_design(
            endog_df=endog_train_std,
            exog_df=exog_train_std,
            p=p,
        )

        Beta = coef.values  # (k, m)
        Yhat_train_std = Xv_train @ Beta
        resid_std = Yv_train - Yhat_train_std  # (T_eff, m)

        # train 当てはめ値を元スケールに戻す
        fitted_train_std = pd.DataFrame(
            Yhat_train_std,
            index=idx_train,
            columns=endog_cols,
        )
        fitted_train = fitted_train_std.copy()
        for col in endog_cols:
            fitted_train[col] = fitted_train_std[col] * stds[col] + means[col]

        # マクロが内生なので，MACRO_COLS に含まれるものだけで RMSE を計算
        macro_in_model = [c for c in endog_cols if c in MACRO_COLS]
        if len(macro_in_model) == 0:
            print("  [注意] このモデルにはマクロ内生変数が含まれていません。RMSEは NaN にします。")

        # train RMSE
        if macro_in_model:
            train_resid_macro = fitted_train[macro_in_model] - train_raw.loc[idx_train, macro_in_model]
            train_rmse_macro = float(np.sqrt((train_resid_macro ** 2).mean().mean()))
        else:
            train_rmse_macro = np.nan

        # --- test RMSE（逐次予測） ---
        pred_std = forecast_varx(
            model=model,
            df_hist_std=train_std[endog_cols + exog_cols],
            df_future_exog_std=test_std[exog_cols] if exog_cols else test_std.iloc[:, 0:0],
            steps=len(test_std),
        )
        pred = pred_std.copy()
        for col in endog_cols:
            pred[col] = pred_std[col] * stds[col] + means[col]

        if macro_in_model:
            test_resid_macro = pred[macro_in_model] - test_raw[macro_in_model]
            test_rmse_macro = float(np.sqrt((test_resid_macro ** 2).mean().mean()))
        else:
            test_rmse_macro = np.nan

        if np.isfinite(train_rmse_macro) and train_rmse_macro > 0:
            overfit_ratio = float(test_rmse_macro / train_rmse_macro)
        else:
            overfit_ratio = np.nan

        print(f"  訓練RMSE（マクロ）: {train_rmse_macro:.4f}  |  テストRMSE（マクロ）: {test_rmse_macro:.4f}")
        print(f"  RMSE比（テスト/訓練）: {overfit_ratio:.3f}")

        # --- 係数の感度（行ブートストラップ） ---
        # ケースBSは重いので今回はスキップ（全部 NaN を入れておく）
        coef_sd_mean_boot = np.nan
        coef_sd_max_boot = np.nan
        sd_const_mean_boot = np.nan
        sd_lag_mean_boot = np.nan
        sd_exog_mean_boot = np.nan
        frob_mean_boot = np.nan
        frob_max_boot = np.nan

        # --- 係数の感度（平滑化ブートストラップ：ランダムノイズ） ---
        (
            Beta_sd_noise_df,
            coef_sd_mean_noise,
            coef_sd_max_noise,
            sd_const_mean_noise,
            sd_lag_mean_noise,
            sd_exog_mean_noise,
            frob_mean_noise,
            frob_max_noise,
        ) = coef_sensitivity_noise(
            df_train_std=train_std[endog_cols + exog_cols],
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=p,
            ridge=ridge,
            base_coef=coef,
            noise_scale=0.2,
            B=200,
            random_state=1,
        )
        print(f"  [ノイズBS] 係数SD平均: {coef_sd_mean_noise:.4f}, "
              f"係数SD最大: {coef_sd_max_noise:.4f}")
        print(f"  [ノイズBS] Frobenius平均: {frob_mean_noise:.4f}, "
              f"最大: {frob_max_noise:.4f}")

        Beta_sd_noise_df.to_csv(
            OUTDIR / f"係数標準偏差_ノイズブートストラップ_{name}.csv",
            encoding="utf-8-sig",
        )

        # --- コンパニオン行列の最大固有値 ---
        max_eig = max_abs_eig_of_companion(A_list, p)
        print(f"  コンパニオン行列の最大|固有値|: {max_eig:.4f}")

        # --- AIC っぽい指標 ---
        n_params = Beta.shape[0]  # 行数 = c + A + B の本数
        aic_like = compute_aic_like(resid_std, n_params)
        print(f"  AIC風指標: {aic_like:.2f}")

        # --- サマリー行を保存（マクロ名は日本語で） ---
        macro_jp_in_model = [JP_MACRO.get(c, c) for c in macro_in_model]

        summary_rows.append({
            "モデル名": name,
            "ラグ次数p": p,
            "リッジ係数": ridge,
            "内生変数の数": len(endog_cols),
            "外生変数の数": len(exog_cols),
            "対象マクロ列（日本語）": ";".join(macro_jp_in_model),
            "訓練RMSE（マクロ）": train_rmse_macro,
            "テストRMSE（マクロ）": test_rmse_macro,
            "RMSE比（テスト/訓練）": overfit_ratio,
            # ケース・ブートストラップ（今回は NaN）
            "ケースBS_係数SD平均": coef_sd_mean_boot,
            "ケースBS_係数SD最大": coef_sd_max_boot,
            "ケースBS_c係数SD平均": sd_const_mean_boot,
            "ケースBS_A係数SD平均": sd_lag_mean_boot,
            "ケースBS_B係数SD平均": sd_exog_mean_boot,
            "ケースBS_Frobenius平均": frob_mean_boot,
            "ケースBS_Frobenius最大": frob_max_boot,
            # ノイズ・ブートストラップ
            "ノイズBS_係数SD平均": coef_sd_mean_noise,
            "ノイズBS_係数SD最大": coef_sd_max_noise,
            "ノイズBS_c係数SD平均": sd_const_mean_noise,
            "ノイズBS_A係数SD平均": sd_lag_mean_noise,
            "ノイズBS_B係数SD平均": sd_exog_mean_noise,
            "ノイズBS_Frobenius平均": frob_mean_noise,
            "ノイズBS_Frobenius最大": frob_max_noise,
            # 安定性 & AIC
            "コンパニオン最大固有値絶対値": max_eig,
            "AIC風指標": aic_like,
        })

        # --- グラフ出力：実績＋当てはめ＋予測（マクロ内生変数） ---
        for col in macro_in_model:
            jp_name = JP_MACRO.get(col, col)
            out_png = OUTDIR / f"{name}_fit_forecast_{jp_name}.png"
            plot_fit_and_forecast(
                df_all=df_model,
                train_end=train_raw.index[-1],
                fitted_train=fitted_train,
                pred_test=pred,
                col=col,
                outpath=out_png,
                jp_name=jp_name,
            )

    # ==========================
    # 結果をまとめて保存
    # ==========================
    summary_df = pd.DataFrame(summary_rows)
    # RMSE比でソート（小さい = 過学習が弱い）順
    summary_df = summary_df.sort_values("RMSE比（テスト/訓練）")

    summary_df.to_csv(
        OUTDIR / "VARXモデル診断サマリー_マクロ内生.csv",
        encoding="utf-8-sig",
        index=False,
    )
    print("\n=== 診断サマリー（日本語列名） ===")
    print(summary_df)
    print("\n[INFO] 診断結果を保存しました:", OUTDIR.resolve())


if __name__ == "__main__":
    main()