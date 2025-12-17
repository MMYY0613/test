import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# =========================
# 設定
# =========================
DATA_PATH = "data/ret_plus_sent_last8q_7d_mani.csv"
TEST_H = 2                  # 最後2点を予測
P = 1                       # VARX(1)
RIDGE_7  = 1e-2             # 7次元: 弱め
RIDGE_16 = 1e0              # 16次元: 強め（データ少ないので）

OUT7  = Path("out_var_7")
OUT16 = Path("out_var_16")

# 7本内生 = リターン6本 + SENT_AVG
RET6_FOR_7 = ["FOODS","ENERGY","MAT_CHEM","AUTO","IT_SERV","BANKS"]

# 8セクターリターン
RET8 = ["FOODS","ENERGY","MAT_CHEM","AUTO","IT_SERV","BANKS","FIN_EX_BANKS","REAL_ESTATE"]

# =========================
# ユーティリティ
# =========================
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, encoding="utf-8-sig")

def save_series(s: pd.Series, path: Path):
    s.to_csv(path, encoding="utf-8-sig", header=True)

def heatmap(A: pd.DataFrame, title: str, outpath: Path, rotation=45):
    plt.figure(figsize=(11, 9))
    vmax = np.nanmax(np.abs(A.values)) if A.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = plt.imshow(A.values, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(A.columns)), A.columns, rotation=rotation, ha="right")
    plt.yticks(range(len(A.index)), A.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_pred_pretty(df_all: pd.DataFrame, train_end_idx, pred_s: pd.Series, col: str, outpath: Path):
    """
    df_all: 学習+テスト全部（dropna後）の実績が入ったDF
    train_end_idx: train.index[-1]
    pred_s: 予測系列（index=test.index）
    """
    actual = df_all[col].astype(float)

    plt.figure(figsize=(9, 4))
    plt.plot(actual.index, actual.values, marker="o", linewidth=1.5, label="actual")
    plt.plot(pred_s.index, pred_s.values, marker="o", linewidth=2.5, label="pred")

    plt.axvline(train_end_idx, linestyle="--", linewidth=1.0)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right")

    plt.title(f"{col}: actual vs pred (split at {train_end_idx.date()})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

# =========================
# VARX(1) Ridge 推定
# =========================
def make_design(endog_df: pd.DataFrame, exog_df: pd.DataFrame | None, p: int):
    Y = endog_df.values
    X_parts = [np.ones((len(endog_df), 1))]  # const
    for i in range(1, p+1):
        X_parts.append(endog_df.shift(i).values)
    if exog_df is not None and exog_df.shape[1] > 0:
        X_parts.append(exog_df.values)
    X = np.concatenate(X_parts, axis=1)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid], endog_df.index[valid]

def fit_varx_ridge(df_train: pd.DataFrame, endog_cols, exog_cols, p=1, ridge=1e-3):
    Y, X, idx = make_design(
        df_train[endog_cols].astype(float),
        df_train[exog_cols].astype(float) if exog_cols else None,
        p
    )

    k = X.shape[1]
    Beta = np.linalg.solve(X.T @ X + ridge*np.eye(k), X.T @ Y)

    labels = (["const"]
              + [f"lag1_{c}" for c in endog_cols]  # p=1前提
              + (exog_cols if exog_cols else []))
    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)

    c = coef.loc["const"].values                               # (m,)
    A1 = coef.loc[[f"lag1_{c}" for c in endog_cols]].values.T  # (m,m)  行=被説明, 列=説明(ラグ)
    B  = coef.loc[exog_cols].values.T if exog_cols else np.zeros((len(endog_cols), 0))

    return {"coef": coef, "c": c, "A1": A1, "B": B,
            "endog": endog_cols, "exog": exog_cols,
            "train_index": idx}

def forecast_varx(model, df_hist: pd.DataFrame, df_future_exog: pd.DataFrame, steps: int):
    endog = model["endog"]
    exog  = model["exog"]
    A1 = model["A1"]
    B  = model["B"]
    c  = model["c"]

    Y_hist = df_hist[endog].copy()
    preds = []

    for h in range(steps):
        y_prev = Y_hist.iloc[-1].values
        x_now  = df_future_exog.iloc[h][exog].values if len(exog) else np.array([])
        y_hat  = c + A1 @ y_prev + (B @ x_now if len(exog) else 0.0)
        preds.append(y_hat)

        Y_hist = pd.concat(
            [Y_hist, pd.DataFrame([y_hat], index=[df_future_exog.index[h]], columns=endog)],
            axis=0
        )

    return pd.DataFrame(preds, index=df_future_exog.index[:steps], columns=endog)

# =========================
# データ準備（マクロが水準なら対数差分）
# =========================
def prep_dataframe(path: str) -> tuple[pd.DataFrame, list[str]]:
    df0 = pd.read_csv(path)

    if "Date" in df0.columns:
        df0["Date"] = pd.to_datetime(df0["Date"])
        df0 = df0.set_index("Date")
    else:
        df0.index = pd.to_datetime(df0.index)

    df0 = df0.sort_index()

    # マクロ整形：あれば水準→対数差分。なければダミー。
    EXOG = []

    if "GDP_LEVEL" in df0.columns and "GDP_LOGDIFF" not in df0.columns:
        df0["GDP_LOGDIFF"] = np.log(df0["GDP_LEVEL"]).diff()
    if "GDP_LOGDIFF" in df0.columns:
        EXOG.append("GDP_LOGDIFF")

    if "NIKKEI_LEVEL" in df0.columns and "NIKKEI_LOGRET" not in df0.columns:
        df0["NIKKEI_LOGRET"] = np.log(df0["NIKKEI_LEVEL"]).diff()
    if "NIKKEI_LOGRET" in df0.columns:
        EXOG.append("NIKKEI_LOGRET")

    if len(EXOG) == 0:
        rng = np.random.default_rng(42)
        df0["GDP_LOGDIFF"]   = rng.normal(0.002, 0.003, size=len(df0))
        df0["NIKKEI_LOGRET"] = rng.normal(0.01, 0.05, size=len(df0))
        EXOG = ["GDP_LOGDIFF", "NIKKEI_LOGRET"]

    df0[EXOG] = df0[EXOG].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df0, EXOG

# =========================
# 1ケース実行して保存
# =========================
def run_and_save(df0: pd.DataFrame, EXOG: list[str], ENDOG: list[str], ret_cols_for_rmse: list[str],
                 outdir: Path, ridge: float, tag: str):
    ensure_dir(outdir)

    # 使う列だけに絞って欠損除去
    dfm = df0[ENDOG + EXOG].copy()
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna()

    if len(dfm) <= (P + TEST_H):
        raise ValueError(f"データ点が少なすぎ: len(dfm)={len(dfm)} (need > P+TEST_H={P+TEST_H})")

    split = max(len(dfm) - TEST_H, 1)
    train = dfm.iloc[:split]
    test  = dfm.iloc[split:]

    # 学習
    model = fit_varx_ridge(train, ENDOG, EXOG, p=P, ridge=ridge)

    # 予測（テスト区間）
    pred = forecast_varx(model, train, test[EXOG], steps=len(test))

    # 係数分解して保存
    A1 = pd.DataFrame(model["A1"], index=ENDOG, columns=ENDOG)
    B  = pd.DataFrame(model["B"],  index=ENDOG, columns=EXOG)
    c  = pd.Series(model["c"], index=ENDOG, name="c")
    coef = model["coef"]

    save_df(coef, outdir / "coef_table.csv")
    save_df(A1,   outdir / "A1.csv")
    save_df(B,    outdir / "B.csv")
    save_series(c, outdir / "c.csv")

    # RMSE（全内生 / リターンだけ）
    rmse_all = np.sqrt(((pred[ENDOG] - test[ENDOG])**2).mean(axis=0))
    rmse_ret = np.sqrt(((pred[ret_cols_for_rmse] - test[ret_cols_for_rmse])**2).mean(axis=0))

    save_series(rmse_all.sort_values(), outdir / "rmse_all.csv")
    save_series(rmse_ret.sort_values(), outdir / "rmse_returns.csv")

    pred_vs_act = pd.concat([pred.add_prefix("pred_"), test.add_prefix("act_")], axis=1)
    save_df(pred_vs_act, outdir / "pred_vs_actual.csv")

    # メタ
    meta = {
        "tag": tag,
        "P": P,
        "TEST_H": TEST_H,
        "ridge": ridge,
        "ENDOG": ENDOG,
        "EXOG": EXOG,
        "ret_cols_for_rmse": ret_cols_for_rmse,
        "n_total": int(len(dfm)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "index": [str(x.date()) for x in dfm.index],
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 図：A1ヒートマップ
    heatmap(A1, f"A1 heatmap ({tag}, ridge={ridge})", outdir / "A1_heatmap.png", rotation=45)

    # 図：予測（見やすい版）
    for col in ret_cols_for_rmse:
        plot_pred_pretty(
            df_all=dfm,
            train_end_idx=train.index[-1],
            pred_s=pred[col],
            col=col,
            outpath=outdir / f"pred_{col}.png"
        )

    print(f"[DONE] {tag} -> {outdir}")
    print("  RMSE returns:\n", rmse_ret.sort_values())
    return {"dfm": dfm, "train": train, "test": test, "pred": pred, "model": model}

# =========================
# 実行
# =========================
df0, EXOG = prep_dataframe(DATA_PATH)

# リターン列の存在確認
ret8_avail = [c for c in RET8 if c in df0.columns]
if len(ret8_avail) < 6:
    raise ValueError(f"returns columns too few: {ret8_avail}")

# センチ列の存在確認
sent8_avail = [f"sent_{c}" for c in ret8_avail if f"sent_{c}" in df0.columns]
if len(sent8_avail) < len(ret8_avail):
    missing = sorted(set([f"sent_{c}" for c in ret8_avail]) - set(sent8_avail))
    raise ValueError(f"missing sentiment columns: {missing}")

# ---- out_var_7：リターン6 + SENT_AVG = 7内生 ----
ret6_avail = [c for c in RET6_FOR_7 if c in df0.columns]
if len(ret6_avail) != 6:
    raise ValueError(f"RET6_FOR_7 not found as 6 cols. got={ret6_avail}")

sent6_cols = [f"sent_{c}" for c in ret6_avail]
df0["SENT_AVG"] = df0[sent6_cols].mean(axis=1)

ENDOG_7 = ret6_avail + ["SENT_AVG"]

run_and_save(
    df0=df0,
    EXOG=EXOG,
    ENDOG=ENDOG_7,
    ret_cols_for_rmse=ret6_avail,   # RMSEはリターン6本だけ
    outdir=OUT7,
    ridge=RIDGE_7,
    tag="ENDOG7 = returns6 + SENT_AVG"
)

# ---- out_var_16：リターン8 + sent8 = 16内生 ----
ENDOG_16 = ret8_avail + [f"sent_{c}" for c in ret8_avail]

run_and_save(
    df0=df0,
    EXOG=EXOG,
    ENDOG=ENDOG_16,
    ret_cols_for_rmse=ret8_avail,   # RMSEはリターン8本
    outdir=OUT16,
    ridge=RIDGE_16,
    tag="ENDOG16 = returns8 + sent8"
)

print("\nAll outputs saved to:")
print(" ", OUT7.resolve())
print(" ", OUT16.resolve())