from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
 
# =========================================================
# matplotlib
# =========================================================
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.family"] = ["Hiragino Sans"]  # Windowsなら適宜変更

# =========================================================
# 共通設定
# =========================================================
DATA_PATH = "./data/all_q_merged.csv"
BASE_OUTDIR = Path("./output_pca_runs")
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

SECTOR_COLS = [
    "RET_FOODS","RET_ENERGY_RESOURCES","RET_CONSTRUCTION_MATERIALS",
    "RET_RAW_MAT_CHEM","RET_PHARMACEUTICAL","RET_AUTOMOBILES_TRANSP_EQUIP",
    "RET_STEEL_NONFERROUS","RET_MACHINERY","RET_ELEC_APPLIANCES_PRECISION",
    "RET_IT_SERV_OTHERS","RET_ELECTRIC_POWER_GAS","RET_TRANSPORT_LOGISTICS",
    "RET_COMMERCIAL_WHOLESALE","RET_RETAIL_TRADE","RET_BANKS",
    "RET_FIN_EX_BANKS","RET_REAL_ESTATE","RET_TEST",
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

JP_LABEL = {
    # "RET_FOODS": "食品",
}

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


# =========================================================
# 便利関数（共通）
# =========================================================
def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def sanitize_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def load_all_q_merged(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.index.isna().any():
        # indexに日時がいない場合は Date列を試す
        df2 = pd.read_csv(path, encoding="utf-8-sig")
        if "Date" not in df2.columns:
            raise ValueError("日時indexでもDate列でも読み取れない。CSV形式を確認して。")
        df2["Date"] = pd.to_datetime(df2["Date"])
        df2 = df2.set_index("Date").sort_index()
        df2 = df2.replace([np.inf, -np.inf], np.nan)
        return df2

    return df


def ensure_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # GDP_LOGDIFF
    if "GDP_LOGDIFF" not in out.columns and "GDP" in out.columns:
        out["GDP_LOGDIFF"] = np.log(out["GDP"]).diff()

    # NIKKEI_LOGRET/TOPIX_LOGRET/FX_LOGRET
    if "NIKKEI_LOGRET" not in out.columns and "NIKKEI" in out.columns:
        out["NIKKEI_LOGRET"] = np.log(out["NIKKEI"]).diff()

    if "TOPIX_LOGRET" not in out.columns and "TOPIX" in out.columns:
        out["TOPIX_LOGRET"] = np.log(out["TOPIX"]).diff()

    if "FX_LOGRET" not in out.columns:
        if "USD_JPY" in out.columns:
            out["FX_LOGRET"] = np.log(out["USD_JPY"]).diff()

    # TANKAN_BUSI
    if "TANKAN_BUSI" not in out.columns:
        if "TANKAN_ACT" in out.columns and "TANKAN_FCST" in out.columns:
            out["TANKAN_BUSI"] = out["TANKAN_ACT"].combine_first(out["TANKAN_FCST"])

    return out


# =========================================================
# PCA（リークなし：trainでfit → testはtransform）
#   - k_use=Noneなら cum_th 到達本数（+max_pcで上限）
#   - k_use=intなら固定k（cum_thは“表示/参考用”）
#   - pc_name_style: "jp" -> セクター_PC1... / "en" -> PC1...
# =========================================================
def pca_no_leak(
    X_all: pd.DataFrame,
    train_start_pos: int,
    train_end_pos: int,     # train: [start:end)
    test_size: int,
    cum_th: float = 0.9,
    k_use: int | None = None,
    max_pc: int | None = None,
    pc_name_style: str = "jp",
):
    n = len(X_all)
    test_size = int(test_size)
    train_start_pos = int(train_start_pos)
    train_end_pos = int(train_end_pos)

    if train_start_pos < 0:
        train_start_pos = 0
    if train_end_pos > n - test_size:
        train_end_pos = n - test_size
    if train_end_pos <= train_start_pos:
        train_end_pos = train_start_pos + 1

    X_train = X_all.iloc[train_start_pos:train_end_pos]
    X_test = X_all.iloc[train_end_pos:train_end_pos + test_size]

    if len(X_train) < 2:
        raise ValueError("trainが短すぎる")
    if len(X_test) < 1:
        raise ValueError("testが短すぎる")

    split_ts = X_all.index[train_end_pos]  # test開始点

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    pca = PCA()
    Ztr_full = pca.fit_transform(Xtr)
    Zte_full = pca.transform(Xte)

    expl = pca.explained_variance_ratio_
    cum = np.cumsum(expl)

    k_auto = int(np.argmax(cum >= cum_th) + 1)

    if k_use is None:
        k = k_auto
        if max_pc is not None:
            k = max(1, min(k, int(max_pc)))
    else:
        k = int(k_use)
        k = max(1, min(k, X_all.shape[1]))

    if pc_name_style == "jp":
        pc_cols = [f"セクター_PC{i+1}" for i in range(k)]
    elif pc_name_style == "en":
        pc_cols = [f"PC{i+1}" for i in range(k)]
    else:
        raise ValueError("pc_name_style must be 'jp' or 'en'")

    pc_train = pd.DataFrame(Ztr_full[:, :k], index=X_train.index, columns=pc_cols)
    pc_test  = pd.DataFrame(Zte_full[:, :k], index=X_test.index,  columns=pc_cols)
    pc_all = pd.concat([pc_train, pc_test], axis=0)

    loading = pd.DataFrame(pca.components_[:k].T, index=X_all.columns, columns=pc_cols)
    pca_info = pd.DataFrame({"寄与率": expl[:k], "累積寄与率": cum[:k]}, index=pc_cols)

    meta = {
        "train_start": X_train.index.min(),
        "train_end": X_train.index.max(),
        "test_start": X_test.index.min(),
        "test_end": X_test.index.max(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "k_auto_by_cum_th": k_auto,
        "k_used": k,
        "cum_th": float(cum_th),
        "cum_at_k_used": float(cum[k-1]),
        "train_start_pos": train_start_pos,
        "train_end_pos": train_end_pos,
        "pc_name_style": pc_name_style,
    }
    return pc_all, loading, pca_info, split_ts, scaler, pca, meta


def plot_pc_timeseries(pc_all: pd.DataFrame, split_ts: pd.Timestamp, title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    pc_all.plot(ax=ax, linewidth=1.6)

    ax.axvline(split_ts, linestyle="--", linewidth=1.5)
    ax.axhline(0.0, linestyle=":", linewidth=1.2)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)

    ax.set_xlim(pc_all.index.min(), pc_all.index.max())
    ax.margins(x=0)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.tick_params(axis="x", which="both", labelbottom=True)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)

    ax.set_title(title)
    ax.set_xlabel("日時")
    ax.set_ylabel("主成分スコア")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_loading_top10(loading: pd.DataFrame, pc_name: str, outpath: Path, jp_label: dict[str, str] | None = None):
    jp_label = jp_label or {}
    top = loading[pc_name].abs().sort_values(ascending=False).head(10).index
    plot_df = loading.loc[top, [pc_name]].copy()
    plot_df.index = [jp_label.get(x, x) for x in plot_df.index]

    plt.figure(figsize=(10, 4))
    ax = plot_df.plot(kind="bar", ax=plt.gca(), legend=False)
    ax.axhline(0.0, linewidth=1.2, linestyle=":")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    plt.title(f"{pc_name} の負荷量（上位10）")
    plt.xlabel("セクター")
    plt.ylabel("負荷量（重み）")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================================================
# ①相当：PCA grid run（train_startもtrain_endも動かす）
# =========================================================
def save_one_run_gridstyle(
    out_dir: Path,
    pc_all: pd.DataFrame,
    loading: pd.DataFrame,
    pca_info: pd.DataFrame,
    split_ts: pd.Timestamp,
    scaler: StandardScaler,
    meta: dict,
    jp_label: dict[str, str],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.Series(meta).to_csv(out_dir / "meta.csv", encoding="utf-8-sig")
    pc_all.to_csv(out_dir / "セクター_主成分スコア.csv", encoding="utf-8-sig")
    loading.to_csv(out_dir / "セクター_負荷量.csv", encoding="utf-8-sig")
    pca_info.to_csv(out_dir / "主成分_寄与率.csv", encoding="utf-8-sig")

    # 線形結合係数（a,b）
    pc_cols = list(pca_info.index)

    a_mat = loading[pc_cols].copy()
    a_mat = a_mat.div(scaler.scale_, axis=0)

    b_vec: dict[str, float] = {}
    for pc_name in pc_cols:
        w = loading[pc_name].values
        b = -np.sum(w * scaler.mean_ / scaler.scale_)
        b_vec[pc_name] = float(b)
    b_ser = pd.Series(b_vec, name="定数項_b")

    a_mat.to_csv(out_dir / "主成分_線形結合係数_a.csv", encoding="utf-8-sig")
    b_ser.to_csv(out_dir / "主成分_線形結合_定数項_b.csv", encoding="utf-8-sig")

    a_with_b = a_mat.copy()
    a_with_b.loc["(定数項)"] = b_ser
    a_with_b.to_csv(out_dir / "主成分_線形結合係数_a_定数b込み.csv", encoding="utf-8-sig")

    # 図：主成分時系列
    title = f"セクター主成分スコアの推移（train fit → test transform, test={meta['test_size']}）"
    plot_pc_timeseries(pc_all, split_ts, title, out_dir / "主成分スコア_時系列.png")

    # 図：負荷量トップ10
    for pc_name in pc_cols:
        pc_idx = pc_cols.index(pc_name) + 1
        plot_loading_top10(loading, pc_name, out_dir / f"負荷量_PC{pc_idx}_上位10.png", jp_label=jp_label)


def run_pca_grid(
    df_raw: pd.DataFrame,
    test_size: int = 4,
    min_train: int = 4,
    cum_th: float = 0.90,
    max_pc: int | None = None,
    max_runs: int | None = 50,
    print_every: int = 50,
    skip_if_exists: bool = False,
):
    # PCAに使う列だけ抜き出して数値化＆欠損除去（index位置を安定させる）
    X = safe_numeric(df_raw[SECTOR_COLS].copy(), SECTOR_COLS).dropna().sort_index()

    n = len(X)
    if n < (min_train + test_size):
        raise ValueError(f"データが足りない: n={n}, MIN_TRAIN+TEST_SIZE={min_train+test_size}")

    # 全パターン数
    total = 0
    for train_end_pos in range(min_train, n - test_size + 1):
        total += (train_end_pos - min_train + 1)

    weight_rows = []
    i = 0

    for train_end_pos in range(min_train, n - test_size + 1):
        for train_start_pos in range(0, train_end_pos - min_train + 1):
            i += 1
            if max_runs is not None and i > max_runs:
                break

            if (i % print_every == 0) or (i == 1) or (i == total) or (max_runs is not None and i == max_runs):
                print(f"[{i}/{total}] {i/total:.1%}  train_pos=[{train_start_pos}:{train_end_pos})  test_pos=[{train_end_pos}:{train_end_pos+test_size})")

            pc_all, loading, pca_info, split_ts, scaler, _, meta = pca_no_leak(
                X_all=X,
                train_start_pos=train_start_pos,
                train_end_pos=train_end_pos,
                test_size=test_size,
                cum_th=cum_th,
                k_use=None,
                max_pc=max_pc,
                pc_name_style="jp",
            )

            train_label = f"{meta['train_start']:%Y-%m}~{meta['train_end']:%Y-%m}"
            test_label  = f"{meta['test_start']:%Y-%m}~{meta['test_end']:%Y-%m}"
            folder_name = f"train[{train_label}]_test[{test_label}]_trainsize={meta['train_size']}"
            out_dir = BASE_OUTDIR / folder_name

            if skip_if_exists and out_dir.exists():
                continue

            save_one_run_gridstyle(
                out_dir=out_dir,
                pc_all=pc_all,
                loading=loading,
                pca_info=pca_info,
                split_ts=split_ts,
                scaler=scaler,
                meta=meta,
                jp_label=JP_LABEL,
            )

            # weightwo：全run×全PC×全セクター負荷量
            for pc_name in pca_info.index:
                row = {
                    "run_dir": out_dir.name,
                    "train_start": meta["train_start"],
                    "train_end": meta["train_end"],
                    "test_start": meta["test_start"],
                    "test_end": meta["test_end"],
                    "train_size": meta["train_size"],
                    "test_size": meta["test_size"],
                    "pc": pc_name,
                    "explained_ratio": float(pca_info.loc[pc_name, "寄与率"]),
                    "cum_ratio": float(pca_info.loc[pc_name, "累積寄与率"]),
                    "cum_th": float(meta["cum_th"]),
                    "cum_at_k_used": float(meta["cum_at_k_used"]),
                }
                for c in SECTOR_COLS:
                    row[c] = float(loading.loc[c, pc_name])
                weight_rows.append(row)

        if max_runs is not None and i >= max_runs:
            break

    weightwo = pd.DataFrame(weight_rows)
    weightwo.to_csv(BASE_OUTDIR / "weightwo.csv", index=False, encoding="utf-8-sig")

    print("DONE(pca_grid). 出力先:", BASE_OUTDIR.resolve())
    print("weightwo:", (BASE_OUTDIR / "weightwo.csv").resolve())
    print("runs_done:", i, " / total_possible:", total)


# =========================================================
# ②相当：diagnostics（PCA固定k + VARX ridge + IRF等）
# =========================================================
def save_pca_outputs_diagstyle(
    out_dir: Path,
    pc_all: pd.DataFrame,
    loading: pd.DataFrame,
    pca_info: pd.DataFrame,
    split_ts: pd.Timestamp,
    scaler: StandardScaler,
    meta: dict
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(meta).to_csv(out_dir / "meta_pca.csv", encoding="utf-8-sig")

    pc_all.to_csv(out_dir / "PC_scores.csv", encoding="utf-8-sig")
    loading.to_csv(out_dir / "PC_loadings.csv", encoding="utf-8-sig")
    pca_info.to_csv(out_dir / "PC_variance.csv", encoding="utf-8-sig")

    title = f"PC score time series (train fit → test transform, test={meta['test_size']})"
    plot_pc_timeseries(pc_all, split_ts, title, out_dir / "PC_scores_timeseries.png")

    for pc in loading.columns:
        top = loading[pc].abs().sort_values(ascending=False).head(10).index
        plot_df = loading.loc[top, [pc]].copy()

        plt.figure(figsize=(10, 4))
        ax = plot_df.plot(kind="bar", ax=plt.gca(), legend=False)
        ax.axhline(0.0, linewidth=1.2, linestyle=":")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        plt.title(f"{pc} loadings (top10)")
        plt.tight_layout()
        plt.savefig(out_dir / f"loadings_{pc}_top10.png", dpi=200)
        plt.close()


def make_design(endog_df: pd.DataFrame, exog_df: pd.DataFrame | None, p: int = 1):
    Y = endog_df.values  # (T,m)
    X_parts = [np.ones((len(endog_df), 1))]

    for lag in range(1, p + 1):
        X_parts.append(endog_df.shift(lag).values)

    if exog_df is not None and exog_df.shape[1] > 0:
        X_parts.append(exog_df.values)

    X = np.concatenate(X_parts, axis=1)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid], endog_df.index[valid]


def fit_varx_ridge(df_train_std: pd.DataFrame,
                   endog_cols: list[str],
                   exog_cols: list[str],
                   p: int = 1,
                   ridge: float = 1.0):
    endog = df_train_std[endog_cols]
    exog = df_train_std[exog_cols] if exog_cols else None
    Y, X, idx = make_design(endog, exog, p=p)

    k = X.shape[1]
    Beta = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ Y)  # (k,m)

    labels: list[str] = ["c"]
    for lag in range(1, p + 1):
        for c in endog_cols:
            labels.append(f"A{lag}_{c}")
    for c in exog_cols:
        labels.append(f"B_{c}")

    coef = pd.DataFrame(Beta, index=labels, columns=endog_cols)
    c_vec = coef.loc["c"].values

    A_list: dict[int, np.ndarray] = {}
    for lag in range(1, p + 1):
        rows = [f"A{lag}_{c}" for c in endog_cols]
        A_list[lag] = coef.loc[rows].values.T  # (m,m)

    if exog_cols:
        rows_B = [f"B_{c}" for c in exog_cols]
        B_mat = coef.loc[rows_B].values.T      # (m,kx)
    else:
        B_mat = np.zeros((len(endog_cols), 0))

    return {
        "coef": coef,
        "A_list": A_list,
        "B": B_mat,
        "c": c_vec,
        "endog": endog_cols,
        "exog": exog_cols,
        "p": p,
        "train_index": idx,
    }


def forecast_varx(model: dict,
                  df_hist_std: pd.DataFrame,
                  df_future_exog_std: pd.DataFrame,
                  steps: int):
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
            y_hat += A_list[lag] @ y_hist.iloc[-lag].values
        if exog:
            y_hat += B @ x_now
        preds.append(y_hat)

        y_hist = pd.concat(
            [y_hist, pd.DataFrame([y_hat], index=[df_future_exog_std.index[h]], columns=endog)],
            axis=0,
        )

    return pd.DataFrame(preds, index=df_future_exog_std.index[:steps], columns=endog)


def build_companion(A_list: dict[int, np.ndarray], p: int) -> np.ndarray:
    m = A_list[1].shape[0]
    comp = np.zeros((m * p, m * p))
    for lag in range(1, p + 1):
        comp[0:m, (lag - 1) * m: lag * m] = A_list[lag]
    if p > 1:
        comp[m:, 0:(p - 1) * m] = np.eye((p - 1) * m)
    return comp


def max_abs_eig_of_companion(A_list: dict[int, np.ndarray], p: int) -> float:
    eigvals = np.linalg.eigvals(build_companion(A_list, p))
    return float(np.max(np.abs(eigvals)))


def compute_aic_like(resid: np.ndarray, n_params: int) -> float:
    T_eff, m = resid.shape
    Sigma = (resid.T @ resid) / T_eff
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return np.nan
    return float(T_eff * logdet + 2.0 * n_params * m)


def plot_fit_forecast_single(
    df_all: pd.DataFrame,
    train_end: pd.Timestamp,
    fitted_train: pd.DataFrame,
    pred_test: pd.DataFrame,
    col: str,
    outpath: Path,
    title: str,
):
    actual = df_all[col]

    plt.figure(figsize=(9, 4))
    plt.plot(actual.index, actual.values, linewidth=1.4, label="actual")
    plt.plot(fitted_train.index, fitted_train[col].values, linestyle="--", linewidth=2.0, label="fit(train)")
    plt.plot(pred_test.index, pred_test[col].values, marker="o", linewidth=2.0, label="forecast(test)")
    plt.axvline(train_end, linestyle=":", linewidth=1.0, label="train_end")

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

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def irf_exog_no_ci(A_list: dict[int, np.ndarray],
                   B: np.ndarray,
                   p: int,
                   steps: int) -> np.ndarray:
    m, kx = B.shape
    C = np.zeros((steps + 1, m, kx), dtype=float)
    C[0] = B.copy()
    for h in range(1, steps + 1):
        acc = np.zeros((m, kx), dtype=float)
        for lag in range(1, p + 1):
            if h - lag >= 0:
                acc += A_list[lag] @ C[h - lag]
        C[h] = acc
    return C


def save_irf_outputs(
    irf_C: np.ndarray,            # (H+1, m, kx) in STD-y units (given +1 STD-x shock)
    endog_cols: list[str],
    exog_cols: list[str],
    sd_y_raw: pd.Series,          # raw y std (train)
    sd_x_raw: pd.Series,          # raw x std (train)
    out_dir: Path,
    steps: int,
    plot_only_gdp: bool = False,  # ★デフォFalseに（全部描く）
    gdp_level_init: float = 500000.0,  # ★追加：GDP初期値
):
    """
    - CSV: 全マクロ反応（rawに戻したもの）※GDPはLOGDIFFのまま
    - 追加CSV/plot: GDP_LOGDIFF があれば、初期値gdp_level_initからのGDPレベル推移も出す
    """
    H = int(steps)
    h = np.arange(H + 1)

    irf_root = out_dir / "irf_exog"
    irf_root.mkdir(parents=True, exist_ok=True)

    # 1σメモ
    sigmas = pd.concat([sd_y_raw.add_prefix("sigma_y__"), sd_x_raw.add_prefix("sigma_x__")])
    sigmas.to_csv(irf_root / "sigmas_train_raw.csv", encoding="utf-8-sig")

    m = len(endog_cols)
    kx = len(exog_cols)

    for j, xname in enumerate(exog_cols):
        sub = irf_root / f"shock_{sanitize_name(xname)}"
        sub.mkdir(parents=True, exist_ok=True)

        # rawに戻す： y_raw = y_std * sd_y
        raw_mat = np.zeros((H + 1, m), dtype=float)
        for i, yname in enumerate(endog_cols):
            raw_mat[:, i] = irf_C[:, i, j] * float(sd_y_raw[yname])

        irf_df = pd.DataFrame(raw_mat, index=h, columns=endog_cols)
        irf_df.index.name = "horizon"
        irf_df.to_csv(sub / "irf_response_all_macros_RAW.csv", encoding="utf-8-sig")

        # plot対象
        if plot_only_gdp:
            targets = ["GDP_LOGDIFF"] if "GDP_LOGDIFF" in endog_cols else []
        else:
            targets = endog_cols

        # 通常プロット（RAW units）
        for ycol in targets:
            y = irf_df[ycol].values
            plt.figure(figsize=(8, 4))
            plt.plot(h, y, marker="o", linewidth=2.0)
            plt.axhline(0.0, linewidth=1.0)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            sx = float(sd_x_raw.get(xname, np.nan))
            sy = float(sd_y_raw.get(ycol, np.nan))
            plt.title(f"IRF: {xname} shock (+1σ raw={sx:.4g}) → {ycol} (σ={sy:.4g}) [RAW units]")
            plt.xlabel("horizon")
            plt.ylabel(f"{ycol} (raw)")
            plt.tight_layout()
            plt.savefig(sub / f"irf_{sanitize_name(ycol)}_RAW.png", dpi=200, bbox_inches="tight")
            plt.close()

        # ★GDPレベル（初期値500000）を追加で出す
        if "GDP_LOGDIFF" in endog_cols:
            g = irf_df["GDP_LOGDIFF"].values  # Δlog(GDP) の反応（RAW units）
            gdp_level = float(gdp_level_init) * np.exp(np.cumsum(g))

            gdp_level_df = pd.DataFrame({"GDP_LEVEL": gdp_level}, index=h)
            gdp_level_df.index.name = "horizon"
            gdp_level_df.to_csv(sub / "irf_GDP_LEVEL_from_init500000.csv", encoding="utf-8-sig")

            plt.figure(figsize=(8, 4))
            plt.plot(h, gdp_level, marker="o", linewidth=2.0)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            sx = float(sd_x_raw.get(xname, np.nan))
            plt.title(f"IRF (LEVEL): {xname} shock (+1σ raw={sx:.4g}) → GDP level (init={gdp_level_init:g})")
            plt.xlabel("horizon")
            plt.ylabel("GDP level")
            plt.tight_layout()
            plt.savefig(sub / "irf_GDP_LEVEL_from_init500000.png", dpi=200, bbox_inches="tight")
            plt.close()

        memo = pd.Series({
            "shock_var": xname,
            "shock_size_in_std_space": 1.0,
            "shock_size_in_raw_space": float(sd_x_raw.get(xname, np.nan)),
            "note": "Model estimated on standardized data. A +1 std shock corresponds to +sigma_x in raw units.",
            "gdp_level_init": float(gdp_level_init),
            "gdp_level_rule": "GDP_h = GDP0 * exp(sum_{t<=h} IRF_GDP_LOGDIFF(t))",
        })
        memo.to_csv(sub / "shock_definition.csv", encoding="utf-8-sig")

# =========================================================
# 係数感度（ノイズBS）
# =========================================================
def coef_sensitivity_noise(
    df_train_std: pd.DataFrame,
    endog_cols: list[str],
    exog_cols: list[str],
    p: int,
    ridge: float,
    base_coef: pd.DataFrame,
    noise_scale: float = 0.2,
    B: int = 50,
    random_state: int | None = 0,
):
    rng = np.random.default_rng(random_state)
    target_cols = endog_cols + exog_cols
    Beta_list: list[np.ndarray] = []

    for _ in range(int(B)):
        df_pert = df_train_std.copy()
        noise = rng.normal(loc=0.0, scale=float(noise_scale), size=df_pert[target_cols].shape)
        df_pert[target_cols] = df_train_std[target_cols].values + noise

        model_b = fit_varx_ridge(
            df_train_std=df_pert,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=p,
            ridge=ridge,
        )
        Beta_list.append(model_b["coef"].values)

    Beta_arr = np.stack(Beta_list, axis=0)
    Beta_sd = Beta_arr.std(axis=0)
    Beta_sd_df = pd.DataFrame(Beta_sd, index=base_coef.index, columns=base_coef.columns)

    coef_sd_mean = float(Beta_sd.mean())
    coef_sd_max = float(Beta_sd.max())

    idx_const = (base_coef.index == "c")
    idx_lag   = base_coef.index.str.startswith("A")
    idx_exog  = base_coef.index.str.startswith("B")

    sd_const_mean = float(Beta_sd[idx_const, :].mean()) if idx_const.any() else np.nan
    sd_lag_mean   = float(Beta_sd[idx_lag, :].mean()) if idx_lag.any() else np.nan
    sd_exog_mean  = float(Beta_sd[idx_exog, :].mean()) if idx_exog.any() else np.nan

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


# =========================================================
# exogの組み合わせ（PC1..PC_USEの全非空部分集合）
# =========================================================
def pc_exog_combinations(pc_cols: list[str]) -> list[list[str]]:
    combos: list[list[str]] = []
    for r in range(1, len(pc_cols) + 1):
        for subset in itertools.combinations(pc_cols, r):
            combos.append(list(subset))
    return combos


# =========================================================
# ② diagnostics：split1回分
# =========================================================
def run_one_split_diagnostics(
    df_raw: pd.DataFrame,
    train_end_pos: int,
    *,
    test_size: int,
    cum_th: float,
    pc_use: int,
    p_lag: int,
    ridge: float,
    irf_steps: int,
    do_noise_bs: bool,
    noise_bs_B: int,
    noise_scale: float,
    plot_all_macros: bool,
    plot_macros: list[str],
    total: int,
    idx_run: int,
):
    # ---------- sector PCA（train_start固定=0、train_end_posのみ動かす、PC固定本数） ----------
    X_sector = safe_numeric(df_raw[SECTOR_COLS].copy(), SECTOR_COLS).dropna().sort_index()
    n = len(X_sector)
    train_end_pos = int(train_end_pos)
    train_end_pos = max(1, min(train_end_pos, n - int(test_size)))

    pc_all, loading, pca_info, split_ts, scaler_pca, _, meta_pca = pca_no_leak(
        X_all=X_sector,
        train_start_pos=0,
        train_end_pos=train_end_pos,
        test_size=test_size,
        cum_th=cum_th,
        k_use=pc_use,          # ★固定
        max_pc=None,
        pc_name_style="en",    # PC1.. にする（diagnostics側）
    )

    train_label = f"{meta_pca['train_start']:%Y-%m}~{meta_pca['train_end']:%Y-%m}"
    test_label  = f"{meta_pca['test_start']:%Y-%m}~{meta_pca['test_end']:%Y-%m}"
    folder_name = f"train[{train_label}]_test[{test_label}]_trainsize={meta_pca['train_size']}"
    out_dir = BASE_OUTDIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{idx_run}/{total}] ({idx_run/total:.1%}) split -> {folder_name}")

    # pca保存（diagnostics用）
    pca_dir = out_dir / "pca"
    save_pca_outputs_diagstyle(pca_dir, pc_all, loading, pca_info, split_ts, scaler_pca, meta_pca)

    # ---------- macro整備 ----------
    df = ensure_macro_columns(df_raw)
    need_cols = MACRO_COLS.copy()
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"MACRO_COLS に必要な列がデータに無い: {missing}")

    df_macro = safe_numeric(df[MACRO_COLS].copy(), MACRO_COLS).dropna().sort_index()

    # PC & macro の共通indexに揃える
    df_pc = pc_all.copy()
    common_idx = df_macro.index.intersection(df_pc.index)
    df_macro = df_macro.loc[common_idx]
    df_pc = df_pc.loc[common_idx]

    # split（時刻基準で切る）
    train_mask = df_macro.index <= meta_pca["train_end"]
    df_train_macro_raw = df_macro.loc[train_mask].copy()
    df_test_macro_raw  = df_macro.loc[~train_mask].copy().head(int(test_size))

    df_train_pc_raw = df_pc.loc[df_train_macro_raw.index].copy()
    df_test_pc_raw  = df_pc.loc[df_test_macro_raw.index].copy()

    if len(df_test_macro_raw) < int(test_size):
        print("  [SKIP] testが不足:", len(df_test_macro_raw))
        return

    diag_root = out_dir / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)

    # exog combos
    pc_cols = [f"PC{i+1}" for i in range(int(pc_use))]
    exog_sets = pc_exog_combinations(pc_cols)

    endog_cols = MACRO_COLS
    plot_cols = endog_cols if plot_all_macros else [c for c in plot_macros if c in endog_cols]

    summary_rows = []

    for exog_cols in exog_sets:
        exog_tag = "_".join(exog_cols)
        model_dir = diag_root / f"exog_{sanitize_name(exog_tag)}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # raw train/test
        train_raw = pd.concat([df_train_macro_raw[endog_cols], df_train_pc_raw[exog_cols]], axis=1)
        test_raw  = pd.concat([df_test_macro_raw[endog_cols],  df_test_pc_raw[exog_cols]], axis=1)

        # 標準化（train基準）
        means = train_raw.mean()
        stds = train_raw.std(ddof=0).replace(0, 1.0)

        train_std = (train_raw - means) / stds
        test_std  = (test_raw  - means) / stds

        # 推定
        model = fit_varx_ridge(
            df_train_std=train_std,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            p=int(p_lag),
            ridge=float(ridge),
        )

        coef = model["coef"]
        coef.to_csv(model_dir / "coef.csv", encoding="utf-8-sig")

        # train当てはめ
        Yv_train, Xv_train, idx_train = make_design(
            endog_df=train_std[endog_cols],
            exog_df=train_std[exog_cols],
            p=int(p_lag),
        )
        Beta = coef.values
        Yhat_train_std = Xv_train @ Beta
        resid_std = Yv_train - Yhat_train_std

        fitted_train_std = pd.DataFrame(Yhat_train_std, index=idx_train, columns=endog_cols)
        fitted_train_raw = fitted_train_std * stds[endog_cols].values + means[endog_cols].values

        actual_train_raw = train_raw.loc[idx_train, endog_cols]
        rmse_train_each = np.sqrt(((fitted_train_raw - actual_train_raw) ** 2).mean(axis=0))
        rmse_train_mean = float(rmse_train_each.mean())

        # test予測
        pred_std = forecast_varx(
            model=model,
            df_hist_std=train_std,
            df_future_exog_std=test_std[exog_cols],
            steps=len(test_std),
        )
        pred_raw = pred_std * stds[endog_cols].values + means[endog_cols].values

        rmse_test_each = np.sqrt(((pred_raw - test_raw[endog_cols]) ** 2).mean(axis=0))
        rmse_test_mean = float(rmse_test_each.mean())

        overfit_ratio = float(rmse_test_mean / rmse_train_mean) if rmse_train_mean > 0 else np.nan

        # 指標
        max_eig = max_abs_eig_of_companion(model["A_list"], int(p_lag))
        n_params = Beta.shape[0]
        aic_like = compute_aic_like(resid_std, n_params)

        # 係数感度（ノイズBS）
        if do_noise_bs:
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
                df_train_std=train_std,
                endog_cols=endog_cols,
                exog_cols=exog_cols,
                p=int(p_lag),
                ridge=float(ridge),
                base_coef=coef,
                noise_scale=float(noise_scale),
                B=int(noise_bs_B),
                random_state=1,
            )
            Beta_sd_noise_df.to_csv(model_dir / "coef_sd_noise_bs.csv", encoding="utf-8-sig")
        else:
            coef_sd_mean_noise = np.nan
            coef_sd_max_noise = np.nan
            sd_const_mean_noise = np.nan
            sd_lag_mean_noise = np.nan
            sd_exog_mean_noise = np.nan
            frob_mean_noise = np.nan
            frob_max_noise = np.nan

        # plot
        plot_dir = model_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for col in plot_cols:
            title = f"{JP_MACRO.get(col, col)} | exog={exog_tag} | p={p_lag} ridge={ridge}"
            plot_fit_forecast_single(
                df_all=pd.concat([df_train_macro_raw, df_test_macro_raw], axis=0),
                train_end=df_train_macro_raw.index.max(),
                fitted_train=fitted_train_raw,
                pred_test=pred_raw,
                col=col,
                outpath=plot_dir / f"fit_forecast_{sanitize_name(col)}.png",
                title=title,
            )

        # IRF
        irf_C = irf_exog_no_ci(model["A_list"], model["B"], p=int(p_lag), steps=int(irf_steps))
        sd_y_raw = stds[endog_cols]
        sd_x_raw = stds[exog_cols]
        save_irf_outputs(
            irf_C=irf_C,
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            sd_y_raw=sd_y_raw,
            sd_x_raw=sd_x_raw,
            out_dir=model_dir,
            steps=int(irf_steps),
            plot_only_gdp=False,          # ★常に全部描く
            gdp_level_init=500000.0,      # ★GDP初期値
        )

        # RMSE保存
        rmse_tbl = pd.DataFrame({
            "rmse_train": rmse_train_each,
            "rmse_test": rmse_test_each,
        })
        rmse_tbl.loc["__mean__"] = [rmse_train_mean, rmse_test_mean]
        rmse_tbl.to_csv(model_dir / "rmse_by_macro.csv", encoding="utf-8-sig")

        # meta保存
        meta = pd.Series({
            "train_start": df_train_macro_raw.index.min(),
            "train_end": df_train_macro_raw.index.max(),
            "test_start": df_test_macro_raw.index.min(),
            "test_end": df_test_macro_raw.index.max(),
            "train_size": len(df_train_macro_raw),
            "test_size": len(df_test_macro_raw),
            "endog_dim": len(endog_cols),
            "exog_dim": len(exog_cols),
            "exog_cols": ";".join(exog_cols),
            "p": int(p_lag),
            "ridge": float(ridge),
            "rmse_train_mean": rmse_train_mean,
            "rmse_test_mean": rmse_test_mean,
            "rmse_ratio_test_over_train": overfit_ratio,
            "max_abs_eig_companion": max_eig,
            "aic_like": aic_like,
            "noise_bs_done": bool(do_noise_bs),
            "noise_bs_B": int(noise_bs_B) if do_noise_bs else 0,
            "noise_bs_scale": float(noise_scale) if do_noise_bs else np.nan,
            "noise_bs_coef_sd_mean": coef_sd_mean_noise,
            "noise_bs_coef_sd_max": coef_sd_max_noise,
            "noise_bs_sd_const_mean": sd_const_mean_noise,
            "noise_bs_sd_lag_mean": sd_lag_mean_noise,
            "noise_bs_sd_exog_mean": sd_exog_mean_noise,
            "noise_bs_frob_mean": frob_mean_noise,
            "noise_bs_frob_max": frob_max_noise,
        })
        meta.to_csv(model_dir / "meta_diagnostics.csv", encoding="utf-8-sig")

        summary_rows.append({
            "split_folder": out_dir.name,
            "exog": exog_tag,
            "p": int(p_lag),
            "ridge": float(ridge),
            "rmse_train_mean": rmse_train_mean,
            "rmse_test_mean": rmse_test_mean,
            "rmse_ratio": overfit_ratio,
            "max_abs_eig": max_eig,
            "aic_like": aic_like,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["rmse_ratio", "rmse_test_mean"])
    summary_df.to_csv(diag_root / "summary_exog_models.csv", encoding="utf-8-sig", index=False)

    best = summary_df.iloc[0]
    print(f"  -> BEST exog={best['exog']}  rmse_test_mean={best['rmse_test_mean']:.4g}  ratio={best['rmse_ratio']:.3g}")


def run_diagnostics_all_splits(
    df_raw: pd.DataFrame,
    *,
    test_size: int = 4,
    cum_th: float = 0.90,
    pc_use: int = 3,
    p_lag: int = 2,
    ridge: float = 1.0,
    irf_steps: int = 12,
    do_noise_bs: bool = False,
    noise_bs_B: int = 50,
    noise_scale: float = 0.2,
    plot_all_macros: bool = False,
    plot_macros: list[str] | None = None,
    start_idx: int | None = None,     # train_end_pos の開始（pos）
    end_idx: int | None = None,       # train_end_pos の終了（pos, inclusive）
    max_splits: int | None = None,    # 途中で止める
):
    plot_macros = plot_macros or ["GDP_LOGDIFF"]

    # sectorが揃う行だけで n を決める（train_end_pos の基準）
    Xtmp = safe_numeric(df_raw[SECTOR_COLS].copy(), SECTOR_COLS).dropna().sort_index()
    n = len(Xtmp)

    min_train = int(test_size)  # 元スクリプト踏襲（最小train=4なら test_size=4でOK）
    last_train_end = n - int(test_size)

    if last_train_end < min_train:
        raise ValueError("splitを作れない（データ長が短い or 欠損多い）。")

    start_pos = min_train if start_idx is None else max(min_train, int(start_idx))
    end_pos = last_train_end if end_idx is None else min(last_train_end, int(end_idx))

    positions = list(range(start_pos, end_pos + 1))
    if max_splits is not None:
        positions = positions[: int(max_splits)]

    total = len(positions)
    if total == 0:
        raise ValueError("指定範囲でsplitが0件。start/end/max_splitsを確認して。")

    for k, train_end_pos in enumerate(positions, start=1):
        run_one_split_diagnostics(
            df_raw=df_raw,
            train_end_pos=train_end_pos,
            test_size=test_size,
            cum_th=cum_th,
            pc_use=pc_use,
            p_lag=p_lag,
            ridge=ridge,
            irf_steps=irf_steps,
            do_noise_bs=do_noise_bs,
            noise_bs_B=noise_bs_B,
            noise_scale=noise_scale,
            plot_all_macros=plot_all_macros,
            plot_macros=plot_macros,
            total=total,
            idx_run=k,
        )

    print("DONE(diagnostics). 出力先:", BASE_OUTDIR.resolve())


# =========================================================
# CLI（pca_grid / diagnostics / all）
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- pca_grid ----
    p1 = sub.add_parser("pca_grid", help="PCAをtrain_start/train_endの全組合せで回して保存 + weightwo.csv")
    p1.add_argument("--data", default=DATA_PATH)
    p1.add_argument("--test-size", type=int, default=4)
    p1.add_argument("--min-train", type=int, default=4)
    p1.add_argument("--cum-th", type=float, default=0.90)
    p1.add_argument("--max-pc", type=int, default=None)
    p1.add_argument("--max-runs", type=int, default=50)
    p1.add_argument("--print-every", type=int, default=50)
    p1.add_argument("--skip-if-exists", action="store_true")

    # ---- diagnostics ----
    p2 = sub.add_parser("diagnostics", help="train_start固定でsplitを回し、PCA固定k+VARX+IRF等を保存")
    p2.add_argument("--data", default=DATA_PATH)
    p2.add_argument("--test-size", type=int, default=4)
    p2.add_argument("--cum-th", type=float, default=0.90)
    p2.add_argument("--pc-use", type=int, default=3)
    p2.add_argument("--p-lag", type=int, default=2)
    p2.add_argument("--ridge", type=float, default=1.0)
    p2.add_argument("--irf-steps", type=int, default=12)
    p2.add_argument("--do-noise-bs", action="store_true")
    p2.add_argument("--noise-bs-B", type=int, default=50)
    p2.add_argument("--noise-scale", type=float, default=0.2)
    p2.add_argument("--plot-all-macros", action="store_true", default=True)  # ★デフォTrue
    p2.add_argument("--plot-macros", nargs="*", default=[])
    p2.add_argument("--start-idx", type=int, default=None)
    p2.add_argument("--end-idx", type=int, default=None)
    p2.add_argument("--max-splits", type=int, default=None)

    # ---- all ----
    p3 = sub.add_parser("all", help="pca_grid と diagnostics を両方実行")
    p3.add_argument("--data", default=DATA_PATH)

    # pca_grid params
    p3.add_argument("--grid-test-size", type=int, default=4)
    p3.add_argument("--grid-min-train", type=int, default=4)
    p3.add_argument("--grid-cum-th", type=float, default=0.90)
    p3.add_argument("--grid-max-pc", type=int, default=None)
    p3.add_argument("--grid-max-runs", type=int, default=50)
    p3.add_argument("--grid-print-every", type=int, default=50)
    p3.add_argument("--grid-skip-if-exists", action="store_true")

    # diagnostics params
    p3.add_argument("--diag-test-size", type=int, default=4)
    p3.add_argument("--diag-cum-th", type=float, default=0.90)
    p3.add_argument("--diag-pc-use", type=int, default=3)
    p3.add_argument("--diag-p-lag", type=int, default=2)
    p3.add_argument("--diag-ridge", type=float, default=1.0)
    p3.add_argument("--diag-irf-steps", type=int, default=12)
    p3.add_argument("--diag-do-noise-bs", action="store_true")
    p3.add_argument("--diag-noise-bs-B", type=int, default=50)
    p3.add_argument("--diag-noise-scale", type=float, default=0.2)
    p3.add_argument("--diag-plot-all-macros", action="store_true")
    p3.add_argument("--diag-plot-macros", nargs="*", default=["GDP_LOGDIFF"])
    p3.add_argument("--diag-start-idx", type=int, default=None)
    p3.add_argument("--diag-end-idx", type=int, default=None)
    p3.add_argument("--diag-max-splits", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "pca_grid":
        df_raw = load_all_q_merged(args.data)
        run_pca_grid(
            df_raw=df_raw,
            test_size=args.test_size,
            min_train=args.min_train,
            cum_th=args.cum_th,
            max_pc=args.max_pc,
            max_runs=args.max_runs,
            print_every=args.print_every,
            skip_if_exists=args.skip_if_exists,
        )
        return

    if args.cmd == "diagnostics":
        df_raw = load_all_q_merged(args.data)
        df_raw = ensure_macro_columns(df_raw)
        run_diagnostics_all_splits(
            df_raw=df_raw,
            test_size=args.test_size,
            cum_th=args.cum_th,
            pc_use=args.pc_use,
            p_lag=args.p_lag,
            ridge=args.ridge,
            irf_steps=args.irf_steps,
            do_noise_bs=args.do_noise_bs,
            noise_bs_B=args.noise_bs_B,
            noise_scale=args.noise_scale,
            plot_all_macros=args.plot_all_macros,
            plot_macros=args.plot_macros,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_splits=args.max_splits,
        )
        return

    if args.cmd == "all":
        df_raw = load_all_q_merged(args.data)
        df_raw = ensure_macro_columns(df_raw)

        run_pca_grid(
            df_raw=df_raw,
            test_size=args.grid_test_size,
            min_train=args.grid_min_train,
            cum_th=args.grid_cum_th,
            max_pc=args.grid_max_pc,
            max_runs=args.grid_max_runs,
            print_every=args.grid_print_every,
            skip_if_exists=args.grid_skip_if_exists,
        )

        run_diagnostics_all_splits(
            df_raw=df_raw,
            test_size=args.diag_test_size,
            cum_th=args.diag_cum_th,
            pc_use=args.diag_pc_use,
            p_lag=args.diag_p_lag,
            ridge=args.diag_ridge,
            irf_steps=args.diag_irf_steps,
            do_noise_bs=args.diag_do_noise_bs,
            noise_bs_B=args.diag_noise_bs_B,
            noise_scale=args.diag_noise_scale,
            plot_all_macros=args.diag_plot_all_macros,
            plot_macros=args.diag_plot_macros,
            start_idx=args.diag_start_idx,
            end_idx=args.diag_end_idx,
            max_splits=args.diag_max_splits,
        )
        return


if __name__ == "__main__":
    main()
