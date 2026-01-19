from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

# ==========================
# フォント（Mac想定）
# ==========================
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.family"] = ["Hiragino Sans"]

# ==========================
# 設定（ここだけ触ればOK）
# ==========================
DATA_PATH = Path("./data/GDP_PCA.csv")
if not DATA_PATH.exists():
    # 念のため：会話アップロード環境
    alt = Path("/mnt/data/GDP_PCA.csv")
    if alt.exists():
        DATA_PATH = alt

OUTDIR = Path("./out_gdp_arx")
OUTDIR.mkdir(parents=True, exist_ok=True)

TEST_H = 4

# CIを締めたいならまずこれ（おすすめ）
USE_PC_LAG = 1       # ★同時点PCは避ける（リーク/不安定化回避）
P = 1                # ★AR次数は軽く（サンプル少ないとP=2は暴れがち）
RIDGE = 30.0         # ★リッジを上げる（係数の暴れを止める）

IRF_STEPS = 8        # ★長いほどCIは太りやすいので短め推奨

# ブートストラップ設定
B_BOOT = 300
CI_LOW, CI_HIGH = 2.5, 97.5

BOOT_METHOD = "wild"   # "wild" 推奨（CIが締まりやすいこと多い）
BLOCK_LEN = 3          # BOOT_METHOD="block" のときだけ使用

# 外生PC（ここは固定）
PC_COLS = ["PC1", "PC2", "PC3"]
Y_COL = "GDP_LOGDIFF"

# ==========================
# 便利：ヒートマップ
# ==========================
def heatmap(df: pd.DataFrame, title: str, outpath: Path):
    A = df.values
    vmax = np.nanmax(np.abs(A)) if A.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(10, 6))
    im = plt.imshow(A, aspect="auto", vmin=-vmax, vmax=vmax, cmap="bwr")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ==========================
# データ読み込み & 目的変数作成
# ==========================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

df[Y_COL] = np.log(df["GDP"]).diff()

# PCラグを作る（0ならそのまま）
if USE_PC_LAG > 0:
    for c in PC_COLS:
        df[f"{c}_L{USE_PC_LAG}"] = df[c].shift(USE_PC_LAG)
    EXOG_COLS = [f"{c}_L{USE_PC_LAG}" for c in PC_COLS]
else:
    EXOG_COLS = PC_COLS

df_model = df[[Y_COL] + EXOG_COLS].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

# ==========================
# 学習/テスト分割
# ==========================
split = max(len(df_model) - TEST_H, 1)
train_raw = df_model.iloc[:split].copy()
test_raw  = df_model.iloc[split:].copy()

# 標準化（train基準）
mu = train_raw.mean()
sd = train_raw.std(ddof=0).replace(0, 1.0)

train_std = (train_raw - mu) / sd
test_std  = (test_raw  - mu) / sd

# ==========================
# ARX（Ridge）
#   y_t = c + a1 y_{t-1} + ... + ap y_{t-p} + b' x_t + u_t
# ==========================
def make_design_arx(y: pd.Series, X: pd.DataFrame, p: int):
    Y = y.values.reshape(-1, 1)
    parts = [np.ones((len(y), 1))]  # const
    for lag in range(1, p + 1):
        parts.append(y.shift(lag).values.reshape(-1, 1))
    parts.append(X.values)
    Z = np.concatenate(parts, axis=1)

    valid = ~np.isnan(Z).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[valid], Z[valid], y.index[valid]

def fit_arx_ridge(train_std: pd.DataFrame, y_col: str, x_cols: list[str], p: int, ridge: float):
    y = train_std[y_col]
    X = train_std[x_cols]
    Y, Z, idx = make_design_arx(y, X, p=p)

    k = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + ridge * np.eye(k), Z.T @ Y)

    labels = ["const"] + [f"lag{lag}_{y_col}" for lag in range(1, p + 1)] + x_cols
    coef = pd.Series(beta.flatten(), index=labels, name="coef_std")

    return {"coef": coef, "p": p, "y_col": y_col, "x_cols": x_cols, "train_index": idx}, (Y, Z, idx)

def forecast_arx(model: dict, hist_std: pd.DataFrame, future_exog_std: pd.DataFrame, steps: int):
    coef: pd.Series = model["coef"]
    p: int = model["p"]
    y_col: str = model["y_col"]
    x_cols: list[str] = model["x_cols"]

    y_hist = hist_std[y_col].copy()
    preds = []

    for h in range(steps):
        y_hat = float(coef["const"])
        for lag in range(1, p + 1):
            y_hat += float(coef[f"lag{lag}_{y_col}"]) * float(y_hist.iloc[-lag])

        x_now = future_exog_std.iloc[h][x_cols].values
        y_hat += float((coef[x_cols].values * x_now).sum())

        preds.append(y_hat)
        y_hist = pd.concat([y_hist, pd.Series([y_hat], index=[future_exog_std.index[h]])])

    return pd.Series(preds, index=future_exog_std.index[:steps], name="pred_std")

# 推定
model, (Yeff, Zeff, idx_eff) = fit_arx_ridge(train_std, Y_COL, EXOG_COLS, p=P, ridge=RIDGE)

# 予測
pred_std = forecast_arx(model, train_std, test_std[EXOG_COLS], steps=len(test_std))
pred = pred_std * sd[Y_COL] + mu[Y_COL]
rmse = float(np.sqrt(((pred - test_raw[Y_COL]) ** 2).mean()))

# ==========================
# 出力：係数、予測、RMSE
# ==========================
model["coef"].to_csv(OUTDIR / "coef_std.csv", encoding="utf-8-sig")
pred.to_csv(OUTDIR / "pred_gdp_logdiff.csv", encoding="utf-8-sig")
test_raw[Y_COL].to_csv(OUTDIR / "actual_gdp_logdiff.csv", encoding="utf-8-sig")
pd.Series({"RMSE": rmse}).to_csv(OUTDIR / "rmse.csv", encoding="utf-8-sig")

print("=== 係数（標準化） ===")
print(model["coef"])
print("\nRMSE:", rmse)
print("出力先:", OUTDIR.resolve())

# 係数表（絶対値順）
coef_tbl = model["coef"].to_frame("coef_std")
coef_tbl["abs"] = coef_tbl["coef_std"].abs()
coef_tbl = coef_tbl.sort_values("abs", ascending=False)
coef_tbl.to_csv(OUTDIR / "coef_table_sorted.csv", encoding="utf-8-sig")

# ==========================
# ヒートマップ①：相関（GDP_LOGDIFF + PC）
# ==========================
corr = df_model.corr()
corr.to_csv(OUTDIR / "corr.csv", encoding="utf-8-sig")
heatmap(corr, "相関ヒートマップ（GDP_LOGDIFF + PC）", OUTDIR / "corr_heatmap.png")

# ==========================
# ヒートマップ②：係数（1行の回帰だけど視覚化）
# ==========================
# 行：GDP_LOGDIFF、列：const + lag + PC
coef_mat = pd.DataFrame([model["coef"].values], index=[Y_COL], columns=model["coef"].index)
heatmap(coef_mat, "係数ヒートマップ（標準化）", OUTDIR / "coef_heatmap.png")
coef_mat.to_csv(OUTDIR / "coef_matrix_std.csv", encoding="utf-8-sig")

# ==========================
# IRF（AR部分の伝播）
# ==========================
def irf_ar_from_init(a: np.ndarray, steps: int, y0: float):
    y = np.zeros(steps + 1)
    y[0] = y0
    p = len(a)
    for h in range(1, steps + 1):
        s = 0.0
        for lag in range(1, p + 1):
            if h - lag >= 0:
                s += a[lag - 1] * y[h - lag]
        y[h] = s
    return y

coef = model["coef"]
a = np.array([coef[f"lag{lag}_{Y_COL}"] for lag in range(1, P + 1)], dtype=float)

# GDP innovation(+1σ) のIRF（標準化）
irf_gdp = irf_ar_from_init(a, IRF_STEPS, y0=1.0)

# PC(+1σ) のIRF（h=0でb、以降ARで伝播）
irf_dict = {"GDP innovation(+1σ)": irf_gdp}
for pc in EXOG_COLS:
    b = float(coef[pc])
    irf_dict[f"{pc}(+1σ)"] = irf_ar_from_init(a, IRF_STEPS, y0=b)

h = np.arange(IRF_STEPS + 1)
irf_df = pd.DataFrame(irf_dict, index=h)
irf_df.index.name = "horizon"
irf_df.to_csv(OUTDIR / "irf_std.csv", encoding="utf-8-sig")

# IRF図
plt.figure(figsize=(9, 4))
for col in irf_df.columns:
    plt.plot(irf_df.index, irf_df[col], marker="o", linewidth=1.6, label=col)
plt.axhline(0.0, linestyle=":", linewidth=1.2)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.xlabel("期（horizon）")
plt.ylabel(f"{Y_COL}（標準化）")
plt.title(f"IRF（ARX: p={P}, PCラグ={USE_PC_LAG}）")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
plt.tight_layout()
plt.savefig(OUTDIR / "irf_std.png", dpi=200)
plt.close()

# ==========================
# ブートストラップCI（wild推奨）
# ==========================
def block_bootstrap_1d(rng, u, n, block_len):
    out = []
    while len(out) < n:
        s = rng.integers(0, n - block_len + 1)
        out.extend(u[s:s+block_len])
    return np.array(out[:n], dtype=float)

def arx_irf_exog(coef: pd.Series, y_col: str, x_cols: list[str], p: int,
                 shock_x: str, steps: int, scale: float = 1.0):
    a = np.array([coef[f"lag{i}_{y_col}"] for i in range(1, p + 1)], dtype=float)
    b = float(coef[shock_x])

    irf = np.zeros(steps + 1, dtype=float)
    irf[0] = b * scale
    for h in range(1, steps + 1):
        s = 0.0
        for i in range(1, p + 1):
            if h - i >= 0:
                s += a[i - 1] * irf[h - i]
        irf[h] = s
    return irf

def bootstrap_irf_ci_arx(
    train_std: pd.DataFrame,
    model: dict,
    y_col: str,
    x_cols: list[str],
    p: int,
    ridge: float,
    resid_std: np.ndarray,   # (T_eff,1)
    steps: int,
    shock_x: str,
    B: int,
    random_state: int,
    scale: float = 1.0,
    method: str = "wild",     # "wild" / "iid" / "block"
    block_len: int = 3,
):
    rng = np.random.default_rng(random_state)

    y = train_std[y_col].values.astype(float)
    X = train_std[x_cols].values.astype(float)
    T = len(y)
    T_eff = resid_std.shape[0]

    u = resid_std[:, 0].copy()
    u = u - u.mean()  # ★センタリング（地味に効く）

    irf_samps = np.zeros((B, steps + 1), dtype=float)

    coef0: pd.Series = model["coef"]

    for b in range(B):
        # 1) 残差を作る
        if method == "wild":
            sign = rng.choice([-1.0, 1.0], size=T_eff)
            u_star = u * sign
        elif method == "block":
            u_star = block_bootstrap_1d(rng, u, T_eff, block_len=block_len)
        else:  # "iid"
            idx = rng.integers(0, T_eff, size=T_eff)
            u_star = u[idx]

        # 2) y* を生成（fixed-design）
        y_star = np.zeros_like(y)
        y_star[:p] = y[:p]

        for i in range(T_eff):
            t = p + i
            y_det = float(coef0["const"])
            for lag in range(1, p + 1):
                y_det += float(coef0[f"lag{lag}_{y_col}"]) * y_star[t - lag]
            y_det += float((coef0[x_cols].values * X[t, :]).sum())
            y_star[t] = y_det + u_star[i]

        # 3) 再推定
        boot_df = train_std.copy()
        boot_df[y_col] = y_star
        model_b, _ = fit_arx_ridge(boot_df, y_col, x_cols, p=p, ridge=ridge)

        # 4) IRF
        irf_samps[b, :] = arx_irf_exog(
            model_b["coef"], y_col=y_col, x_cols=x_cols, p=p,
            shock_x=shock_x, steps=steps, scale=scale
        )

    lo = np.percentile(irf_samps, CI_LOW, axis=0)
    hi = np.percentile(irf_samps, CI_HIGH, axis=0)
    return lo, hi, irf_samps

def plot_irf_with_ci(h, irf, lo, hi, title, outpath: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(h, irf, marker="o", linewidth=2.0, label="IRF")
    plt.fill_between(h, lo, hi, alpha=0.2, label=f"{CI_LOW:.1f}-{CI_HIGH:.1f}% CI")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("期（horizon）")
    plt.ylabel(f"{Y_COL}（標準化）")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

# 残差（標準化）
beta_vec = model["coef"].values.reshape(-1, 1)
yhat_eff = Zeff @ beta_vec
resid_std = (Yeff - yhat_eff)

# PC1/2/3 のCIつきIRF
for shock_x in EXOG_COLS:
    irf0 = arx_irf_exog(coef, y_col=Y_COL, x_cols=EXOG_COLS, p=P, shock_x=shock_x, steps=IRF_STEPS, scale=1.0)
    lo, hi, _ = bootstrap_irf_ci_arx(
        train_std=train_std,
        model=model,
        y_col=Y_COL,
        x_cols=EXOG_COLS,
        p=P,
        ridge=RIDGE,
        resid_std=resid_std,
        steps=IRF_STEPS,
        shock_x=shock_x,
        B=B_BOOT,
        random_state=42,
        scale=1.0,
        method=BOOT_METHOD,
        block_len=BLOCK_LEN,
    )

    out = pd.DataFrame({"irf": irf0, "ci_low": lo, "ci_high": hi}, index=h)
    out.index.name = "horizon"
    out.to_csv(OUTDIR / f"irf_{shock_x}_to_gdp_std_ci.csv", encoding="utf-8-sig")

    plot_irf_with_ci(
        h=h, irf=irf0, lo=lo, hi=hi,
        title=f"{shock_x} に +1σショック → {Y_COL}（標準化, {BOOT_METHOD} bootstrap）",
        outpath=OUTDIR / f"irf_{shock_x}_to_gdp_ci.png",
    )

print("[DONE] 係数・予測・RMSE・相関/係数ヒートマップ・IRF/CI を出力:", OUTDIR.resolve())

# ==========================
# 実績 vs 予測 図
# ==========================
plt.figure(figsize=(9, 4))
plt.plot(df_model.index, df_model[Y_COL], marker="o", linewidth=1.2, label="実績（GDP_LOGDIFF）")
plt.axvline(train_raw.index[-1], linestyle="--", linewidth=1.0, label="学習終了")
plt.plot(pred.index, pred.values, marker="o", linewidth=2.0, label="予測（test）")

ax = plt.gca()
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=90)

plt.title(f"GDP成長率（log差分）をPCで予測（ARX, p={P}, PCラグ={USE_PC_LAG}, ridge={RIDGE}, test={TEST_H}）")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "pred_plot.png", dpi=200)
plt.close()