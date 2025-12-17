import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/ret_plus_sent_last8q_7d_mani.csv"   # ←添付CSVを置いたパス
DATE_COL  = "Date"                                   # 日付列名（なければ index が日付想定）

# 8セクター（リターン列）
ENDOG_RET_CAND = ["FOODS","ENERGY","MAT_CHEM","AUTO","IT_SERV","BANKS","FIN_EX_BANKS","REAL_ESTATE"]

# マクロ“レベル”列（期末値など）候補：手元の列名に合わせてここを追加/修正
GDP_LEVEL_CAND   = ["GDP", "GDP_LEVEL", "GDP_JP", "GDP_NOMINAL", "GDP_REAL"]
NIKKEI_LEVEL_CAND= ["NIKKEI", "NIKKEI_LEVEL", "NIKKEI225", "NIKKEI_225", "N225"]

# 使うマクロの変換：対数差分（log diff）
USE_LOGDIFF_FOR_GDP    = True   # GDPもlog差分にする（成長率の近似）
USE_LOGDIFF_FOR_NIKKEI = True   # 日経はlog差分＝リターン

# センチを exog に入れる（まずは平均1本）
USE_SENT_AVG_AS_EXOG = True

# p（ラグ次数）
P_LAG = 1

# テスト点数（最後2点を予測に使う）
N_TEST = 2

# Ridge（特異で不安なら 1e-6 とか）
RIDGE = 0.0

# 図の出力先
OUT_DIR = Path("out_varx")
FIG_DIR = OUT_DIR / "fig"
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# helpers
# ============================================================
def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_logdiff(s: pd.Series) -> pd.Series:
    """log差分。0や負があれば NaN にしてから diff する。"""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0, np.nan)
    return np.log(s).diff()

def make_design(df_y: pd.DataFrame, df_x: pd.DataFrame | None, p: int):
    y = df_y.copy()
    x = df_x.copy() if df_x is not None and df_x.shape[1] > 0 else None

    X_parts = [np.ones((len(y), 1))]  # const
    for i in range(1, p + 1):
        X_parts.append(y.shift(i).values)
    if x is not None:
        X_parts.append(x.values)

    X = np.concatenate(X_parts, axis=1)
    Y = y.values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid], y.index[valid]

def fit_varx_ols(df: pd.DataFrame, endog_cols, exog_cols, p: int = 1, ridge: float = 0.0):
    df_y = df[endog_cols].astype(float)
    df_x = df[exog_cols].astype(float) if exog_cols else None
    Y, X, idx = make_design(df_y, df_x, p)

    if ridge > 0:
        k = X.shape[1]
        Beta = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ Y)
    else:
        Beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

    Yhat = X @ Beta
    E = Y - Yhat
    n, k = X.shape
    Sigma = (E.T @ E) / max(n - k, 1)

    labels = ["const"] + [f"lag{i}_{c}" for i in range(1, p + 1) for c in endog_cols] + (exog_cols if exog_cols else [])
    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)

    return {
        "p": p, "beta": Beta, "sigma": Sigma, "train_index": idx,
        "endog_cols": endog_cols, "exog_cols": exog_cols,
        "beta_labels": labels, "coef": coef
    }

def forecast_varx(model, df_hist: pd.DataFrame, df_x_future: pd.DataFrame, steps: int):
    p = model["p"]
    ycols = model["endog_cols"]
    xcols = model["exog_cols"]
    Beta  = model["beta"]

    cur_y = df_hist[ycols].copy()
    preds = []

    for h in range(steps):
        parts = [np.ones((1, 1))]
        for i in range(1, p + 1):
            parts.append(cur_y.iloc[-i].values.reshape(1, -1))
        if xcols:
            parts.append(df_x_future.iloc[h][xcols].values.reshape(1, -1))
        Xrow = np.concatenate(parts, axis=1)

        y_pred = (Xrow @ Beta).reshape(-1)
        preds.append(y_pred)

        nxt = pd.DataFrame([y_pred], columns=ycols, index=[df_x_future.index[h]])
        cur_y = pd.concat([cur_y, nxt], axis=0)

    return pd.DataFrame(preds, index=df_x_future.index[:steps], columns=ycols)

def save_heatmap(A: pd.DataFrame, outpath: Path, title: str):
    vmax = np.nanmax(np.abs(A.values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(A.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(A.shape[1]))
    ax.set_yticks(np.arange(A.shape[0]))
    ax.set_xticklabels(A.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(A.index, fontsize=10)

    ax.set_xticks(np.arange(-.5, A.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, A.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient", rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_pred_vs_act(pred: pd.DataFrame, act: pd.DataFrame, outdir: Path):
    outdir.mkdir(exist_ok=True, parents=True)
    idx = pred.index

    for c in pred.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(idx, act[c].values, marker="o", linewidth=2, label="Actual")
        ax.plot(idx, pred[c].values, marker="o", linewidth=2, label="Pred")

        ax.set_title(f"Pred vs Actual: {c}", fontsize=13, pad=10)
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.3)

        ax.set_xticks(idx)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in idx], rotation=45, ha="right")
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(outdir / f"pred_vs_act_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# ============================================================
# 0) load
# ============================================================
df0 = pd.read_csv(DATA_PATH)

if DATE_COL in df0.columns:
    df0[DATE_COL] = pd.to_datetime(df0[DATE_COL])
    df0 = df0.set_index(DATE_COL)
else:
    df0.index = pd.to_datetime(df0.index)

df0 = df0.sort_index()

# endog（リターン）
ENDOG = [c for c in ENDOG_RET_CAND if c in df0.columns]
if not ENDOG:
    raise ValueError(f"ENDOG returns not found. columns={df0.columns.tolist()}")

# ============================================================
# 1) macro: GDP / Nikkei level -> log diff
# ============================================================
EXOG = []

gdp_col = pick_first_existing(df0, GDP_LEVEL_CAND)
if gdp_col is not None:
    if USE_LOGDIFF_FOR_GDP:
        df0["GDP_LOGDIFF"] = safe_logdiff(df0[gdp_col])
        EXOG.append("GDP_LOGDIFF")
    else:
        df0["GDP_PCT"] = pd.to_numeric(df0[gdp_col], errors="coerce").pct_change()
        EXOG.append("GDP_PCT")

nikkei_col = pick_first_existing(df0, NIKKEI_LEVEL_CAND)
if nikkei_col is not None:
    if USE_LOGDIFF_FOR_NIKKEI:
        df0["NIKKEI_LOGRET"] = safe_logdiff(df0[nikkei_col])
        EXOG.append("NIKKEI_LOGRET")
    else:
        df0["NIKKEI_PCT"] = pd.to_numeric(df0[nikkei_col], errors="coerce").pct_change()
        EXOG.append("NIKKEI_PCT")

# （もしマクロ列が無かったら、最低限動くようにダミーを入れる）
if len(EXOG) == 0:
    rng = np.random.default_rng(42)
    df0["GDP_LOGDIFF"] = pd.Series(rng.normal(0.0, 0.01, size=len(df0)), index=df0.index)
    df0["NIKKEI_LOGRET"] = pd.Series(rng.normal(0.0, 0.03, size=len(df0)), index=df0.index)
    EXOG = ["GDP_LOGDIFF", "NIKKEI_LOGRET"]

# 無限大→NaN
df0[EXOG] = df0[EXOG].replace([np.inf, -np.inf], np.nan)

# ============================================================
# 2) sentiment: sent_* -> average 1本（exog）
# ============================================================
sent_cols = [f"sent_{c}" for c in ENDOG if f"sent_{c}" in df0.columns]
if USE_SENT_AVG_AS_EXOG and len(sent_cols) > 0:
    df0["SENT_AVG"] = df0[sent_cols].mean(axis=1)
    EXOG = EXOG + ["SENT_AVG"]

# ============================================================
# 3) build train/test
# ============================================================
use_cols = ENDOG + EXOG
df = df0[use_cols].copy().dropna()

if len(df) < (P_LAG + 3):
    raise ValueError(f"Too few rows after dropna: {len(df)}. Check missing macro/sent columns.")

split = max(len(df) - N_TEST, 1)
train = df.iloc[:split]
test  = df.iloc[split:]

# ============================================================
# 4) fit + forecast
# ============================================================
model = fit_varx_ols(train, ENDOG, EXOG, p=P_LAG, ridge=RIDGE)
pred  = forecast_varx(model, train, test[EXOG], steps=len(test))

rmse = np.sqrt(((pred - test[ENDOG]) ** 2).mean(axis=0))
print("RMSE by sector:")
print(rmse.sort_values())

print("\nPred vs Actual:")
print(pd.concat([pred.add_prefix("pred_"), test[ENDOG].add_prefix("act_")], axis=1))

# ============================================================
# 5) extract A, B, c  (p=1想定)
# ============================================================
coef = model["coef"]  # rows=regressors, cols=equations(=ENDOG)

# c（定数項ベクトル）
c_vec = coef.loc["const"].copy()  # index=ENDOG

# A1（波及行列）：行=影響を受ける側（式）、列=影響を与える側（ラグ変数）
lag_rows = [f"lag1_{c}" for c in ENDOG]
A1 = coef.loc[lag_rows].T
A1.columns = ENDOG

# B（外生の係数行列）：行=ENDOG, 列=EXOG
B = coef.loc[EXOG].T if len(EXOG) > 0 else pd.DataFrame(index=ENDOG)

# 保存
A1.to_csv(OUT_DIR / "A1.csv", encoding="utf-8-sig")
B.to_csv(OUT_DIR / "B.csv",  encoding="utf-8-sig")
c_vec.to_frame("const").to_csv(OUT_DIR / "c.csv", encoding="utf-8-sig")
pred.to_csv(OUT_DIR / "pred.csv", encoding="utf-8-sig")
test[ENDOG].to_csv(OUT_DIR / "actual.csv", encoding="utf-8-sig")
rmse.to_csv(OUT_DIR / "rmse.csv", encoding="utf-8-sig")

print("\nSaved CSVs to:", OUT_DIR)

# ============================================================
# 6) visualization (savefig only)
# ============================================================
save_heatmap(A1, FIG_DIR / "A1_heatmap.png", title="VARX A1 (lag=1) Spillover Matrix")
if len(EXOG) > 0:
    save_heatmap(B, FIG_DIR / "B_heatmap.png", title="VARX B (exogenous) Coefficients")
save_pred_vs_act(pred, test[ENDOG], FIG_DIR)

print("Saved figs to:", FIG_DIR)