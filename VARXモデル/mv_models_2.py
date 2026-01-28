import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import itertools
import warnings
import seaborn as sns
from scipy.optimize import lsq_linear
import matplotlib.dates as mdates

warnings.simplefilter("ignore")

# =========================================================
# 1. 設定・定数
# =========================================================
CONFIG = {
    "data_path": "./data/all_q_merged_tmp2.csv",
    "output_dir": "./output_pca_final_tmp2_2",
    "train_range": ("2015-01-01", "2016-06-30"),
    "test_steps": 4,
    "pc_max": 3,          # ここは「モデルに入れるPC数」（固定で3）
    "p_lag": 1,
    "ridge": 1.0,
    "do_irf": True,
    "use_constraints": True,
}

BASE_LEVELS = {
    "GDP": 500000,
    "NIKKEI": 20000,
    "USD_JPY": 110,
    "UNEMP_RATE": 3.0,
    "JGB_1Y": 0.1,
    "JGB_2Y": 0.2,
    "JGB_3Y": 0.3,
    "CPI": 100,
}

TARGET_MACRO = list(BASE_LEVELS.keys())

SECTOR_COLS = [
    "RET_FOODS", "RET_ENERGY_RESOURCES", "RET_CONSTRUCTION_MATERIALS", "RET_RAW_MAT_CHEM",
    "RET_PHARMACEUTICAL", "RET_AUTOMOBILES_TRANSP_EQUIP", "RET_STEEL_NONFERROUS",
    "RET_MACHINERY", "RET_ELEC_APPLIANCES_PRECISION", "RET_IT_SERV_OTHERS",
    "RET_ELECTRIC_POWER_GAS", "RET_TRANSPORT_LOGISTICS", "RET_COMMERCIAL_WHOLESALE",
    "RET_RETAIL_TRADE", "RET_BANKS", "RET_FIN_EX_BANKS", "RET_REAL_ESTATE", "RET_TEST",
]

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["Hiragino Sans"]


# =========================================================
# 2. ロジック関数
# =========================================================
def smart_transform(series, name):
    if (series <= 0).any() or "JGB" in name or "UNEMP" in name:
        return series.diff(), "DIFF"
    return np.log(series).diff(), "LOGDIFF"


def prepare_aligned(df_raw):
    m_work = pd.DataFrame(index=df_raw.index)
    meta = {}
    for col in TARGET_MACRO:
        if col in df_raw.columns:
            ts, method = smart_transform(df_raw[col], col)
            m_work[f"{col}_{method}"] = ts
            meta[f"{col}_{method}"] = {"orig": col, "method": method}

    m_df = m_work.dropna()

    s_raw = df_raw[[c for c in SECTOR_COLS if c in df_raw.columns]].interpolate(limit_direction="both")
    # もともとRET_群なら diff しない（あなたの方針を踏襲）
    s_diff = s_raw.dropna() if all(c.startswith("RET_") for c in s_raw.columns) else s_raw.diff().dropna()

    common = s_diff.index.intersection(m_df.index)
    return s_diff.loc[common], m_df.loc[common], common, meta


def make_design(endog, exog, p):
    Y = endog.values
    X_parts = [np.ones((len(endog), 1))]
    for lag in range(1, p+1):
        X_parts.append(endog.shift(lag).values)
    if exog is not None and not exog.empty:
        X_parts.append(exog.values)

    X = np.concatenate(X_parts, axis=1)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)

    # ★追加：有効な行の日時indexも返す
    idx_valid = endog.index[valid]

    return Y[valid], X[valid], idx_valid


def decide_pc_signs_by_groups(pca, sector_df, sector_groups, top_n_for_sign=5, n_fix=3):
    """
    PCAの符号を「指定セクター群の有力(絶対値上位)負荷量の平均が正」になるように決める。
    戻り値: fixed_signs (len=n_fix)
    """
    comps = pca.components_.copy()  # (n_components, n_features)
    n_fix = min(n_fix, comps.shape[0])
    fixed_signs = np.ones(n_fix)

    for i in range(n_fix):
        pc_name = f"PC{i+1}"
        if pc_name not in sector_groups:
            continue

        current_loadings = pd.Series(comps[i], index=sector_df.columns)
        group = sector_groups[pc_name]
        group_loadings = current_loadings[current_loadings.index.isin(group)]
        if group_loadings.empty:
            continue

        top_idx = group_loadings.abs().sort_values(ascending=False).head(top_n_for_sign).index
        sign_check = group_loadings[top_idx].mean()

        if sign_check < 0:
            fixed_signs[i] = -1

    return fixed_signs


def analyze_pca_details(pca, sector_df, pc_cols, root_dir, t_start, t_end, t_size, fixed_signs=None):
    """
    PCA分析：標準化データからのスコア算出・プロット・負荷量CSV出力
    ★fixed_signs を渡すと「モデル投入と同じ符号系」でレポートも統一
    """
    pca_dir = root_dir / "pca_analysis"
    pca_dir.mkdir(exist_ok=True)

    # 訓練+テスト期間のデータを抽出して標準化
    X_target = sector_df.iloc[t_start : t_end + t_size]
    sc = StandardScaler()
    sc.fit(sector_df.iloc[t_start:t_end])
    X_scaled = sc.transform(X_target)

    # 主成分スコアの算出（raw score）
    scores = pca.transform(X_scaled)
    score_df = pd.DataFrame(scores, index=X_target.index, columns=pc_cols)

    expl = pca.explained_variance_ratio_
    components = pca.components_.copy()

    # ★外部から符号固定が来たら、それで統一（ここが本丸）
    if fixed_signs is not None:
        n_fix = min(len(fixed_signs), components.shape[0], score_df.shape[1])
        for i in range(n_fix):
            if fixed_signs[i] < 0:
                components[i] *= -1
                score_df.iloc[:, i] *= -1

    # スコアのCSV出力
    score_df.to_csv(pca_dir / "主成分スコア_生データ.csv", encoding="utf-8-sig")

    # 負荷量（Loadings）のCSV出力
    loadings = pd.DataFrame(components.T, index=sector_df.columns, columns=pc_cols)
    loadings.to_csv(pca_dir / "セクターの負荷量_一覧.csv", encoding="utf-8-sig")

    # 寄与率の保存
    expl_df = pd.DataFrame(expl.reshape(1, -1), columns=pc_cols, index=["ExplainedVariance"])
    expl_df.to_csv(pca_dir / "主成分_寄与率.csv", encoding="utf-8-sig")

    # スコア推移プロット（PC3まで）
    plt.figure(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pc in enumerate(pc_cols[:3]):
        plt.plot(score_df.index, score_df[pc], label=f"{pc} (寄与:{expl[i]*100:.1f}%)", color=colors[i], lw=2)
    plt.title("主成分スコアの推移 (セクター標準化後の生値)")
    plt.xlabel("日付")
    plt.ylabel("主成分スコア")
    plt.axhline(0, color="black", lw=1)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(pca_dir / "PCA_スコア推移_生値.png")
    plt.close()

    # 負荷量の棒グラフ（PC3まで）
    for i, pc in enumerate(pc_cols[:3]):
        plt.figure(figsize=(10, 5))
        loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(10).plot(
            kind="bar", color=colors[i]
        )
        plt.title(f"{pc} セクターの負荷量 (寄与:{expl[i]*100:.1f}%)")
        plt.xlabel("セクター")
        plt.ylabel("負荷量")
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(pca_dir / f"負荷量_{pc}_TOP10.png")
        plt.close()

    print(f"✅ PCA分析完了: {pca_dir}")


def plot_abs_heatmaps(full_pca_df, output_root):
    for pc in ["PC1", "PC2", "PC3"]:
        target_df = full_pca_df[full_pca_df["PC_Type"] == pc]
        if target_df.empty:
            continue

        plot_data = target_df.drop(columns=["PC_Type", "Window", "Explained_Variance"]).T
        plot_data.columns = target_df["Window"]

        plt.figure(figsize=(14, 9))
        sns.heatmap(
            plot_data,
            cmap="Reds",
            annot=False,
            cbar_kws={"label": f"{pc} 負荷量（絶対値）"},
            vmin=0,
            vmax=max(0.5, plot_data.max().max()),
        )

        mean_var = target_df["Explained_Variance"].mean() * 100
        plt.title(f"{pc} 構成セクターの負荷量推移（絶対値）\n[平均寄与率: {mean_var:.1f}%]", fontsize=14)
        plt.xlabel("テストウィンドウ (Window)", fontsize=12)
        plt.ylabel("セクター", fontsize=12)

        plt.tight_layout()
        plt.savefig(Path(output_root) / f"全期間_{pc}_構造推移_絶対値.png")
        plt.close()


def plot_final_boxplots(merged_summary, output_root):
    output_root = Path(output_root)
    plot_df_base = merged_summary.copy().reset_index(drop=True)

    targets = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]

    for col in targets:
        if col not in plot_df_base.columns:
            continue

        plt.figure(figsize=(14, 8))
        plot_data = plot_df_base[plot_df_base[col] != 0].copy() if col != "AIC" else plot_df_base.copy()
        if plot_data.empty:
            continue

        order = plot_data.groupby("モデル構成")[col].mean().sort_values().index

        sns.boxplot(
            data=plot_data,
            x="モデル構成",
            y=col,
            order=order,
            palette="Set3",
            width=0.6,
            # whis=[10, 90],
            boxprops=dict(alpha=0.7),
        )
        sns.pointplot(
            data=plot_data,
            x="モデル構成",
            y=col,
            order=order,
            join=False,
            color="red",
            marker="D",
            scale=0.7,
            errorbar="sd",
            capsize=0.15,
            label="平均 ± 標準偏差",
        )

        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col}の箱ひげ図", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(output_root / f"全期間_箱ひげ_{col}.png", dpi=300)
        plt.close()


def aggregate_results(output_root):
    output_root = Path(output_root)
    all_summaries = []
    pca_abs_details = []

    for window_path in sorted(output_root.glob("Window_Test_*")):
        window_name = window_path.name.replace("Window_Test_", "")

        summary_file = window_path / "モデル比較サマリー.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df["Test_Window"] = window_name
            all_summaries.append(df)

        loadings_file = window_path / "pca_analysis" / "セクターの負荷量_一覧.csv"
        variance_file = window_path / "pca_analysis" / "主成分_寄与率.csv"
        if loadings_file.exists() and variance_file.exists():
            ld_df = pd.read_csv(loadings_file, index_col=0)
            vr_df = pd.read_csv(variance_file, index_col=0)
            for pc in ["PC1", "PC2", "PC3"]:
                if pc in ld_df.columns:
                    abs_row = ld_df[pc].abs().to_frame().T
                    abs_row.index = [f"{window_name}_{pc}"]
                    abs_row.insert(0, "Window", window_name)
                    abs_row.insert(1, "PC_Type", pc)
                    abs_row.insert(2, "Explained_Variance", vr_df.at["ExplainedVariance", pc])
                    pca_abs_details.append(abs_row)

    if not all_summaries:
        return

    merged_summary = pd.concat(all_summaries, ignore_index=True)
    plot_final_boxplots(merged_summary, output_root)

    target_cols = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]
    stats_list = []
    for col in target_cols:
        if col in merged_summary.columns:
            temp = merged_summary.copy()
            if col != "AIC":
                temp[col] = temp[col].replace(0, np.nan)

            res = temp.groupby("モデル構成")[col].agg(
                [(f"{col}_平均", "mean"), (f"{col}_標準偏差", "std"), (f"{col}_最大", "max"), (f"{col}_最小", "min")]
            )
            stats_list.append(res)

    model_perf_detail = pd.concat(stats_list, axis=1)
    model_perf_detail["有効試行回数"] = merged_summary.groupby("モデル構成")["予測RMSE"].apply(lambda x: (x != 0).sum())
    model_perf_detail = model_perf_detail.round(4)
    model_perf_detail.to_csv(output_root / "全ウィンドウ集計_モデル評価_詳細版.csv", encoding="utf-8-sig")

    if pca_abs_details:
        full_pca_df = pd.concat(pca_abs_details)
        full_pca_df.to_csv(output_root / "全ウィンドウ集計_PCA構造_絶対値詳細.csv", encoding="utf-8-sig")
        plot_abs_heatmaps(full_pca_df, output_root)

    print("✅ すべての集計・画像保存（ヒートマップ、箱ひげ図、統計詳細）が完了しました。")


def visualize_model_performance(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    output_dir = Path(csv_path).parent
    plt.style.use("ggplot")

    if "AIC_平均" in df.columns and "予測RMSE_平均" in df.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df,
            x="AIC_平均",
            y="予測RMSE_平均",
            size="予測RMSE_標準偏差",
            hue=df.index,
            sizes=(100, 1000),
            alpha=0.6,
        )

        for i, txt in enumerate(df.index):
            plt.annotate(
                txt,
                (df["AIC_平均"].iloc[i], df["予測RMSE_平均"].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.title("モデルの複雑さ(AIC) vs 予測精度(RMSE)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "全期間_AIC_vs_RMSE_分布.png", dpi=300)
        plt.close()

    print(f"💾 散布図1種を {output_dir} に保存しました。")


def plot_connected_forecasts(output_root, df_raw, meta):
    plt.style.use('default') 
    plt.rcParams['font.family'] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
    
    output_root = Path(output_root)
    m_cols_transformed = list(meta.keys())
    
    all_preds = {m: {p: [] for p in range(4)} for m in m_cols_transformed}

    window_dirs = sorted(output_root.glob("Window_Test_*"))
    
    for i, window_path in enumerate(window_dirs):
        phase = i % 4
        for sub in window_path.iterdir():
            if not sub.is_dir(): continue
            for m_key, m_info in meta.items():
                orig_name = m_info["orig"]
                
                # 1. 予測値の読み込み
                target_csv = sub / f"予測値_水準ベース_{orig_name}.csv"
                if target_csv.exists():
                    pdf = pd.read_csv(target_csv, index_col=0, parse_dates=True)
                    all_preds[m_key][phase].append(pdf[f"{orig_name}_Pred"])

    for m_col, m_info in meta.items():
        orig_name = m_info["orig"]
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        
        # --- 描画2: 実績値（黒） ---
        ax.plot(df_raw.index, df_raw[orig_name], color='black', lw=2, label="実績値", zorder=2)
        
        # --- 描画3: 各フェーズの予測値（点線） ---
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        for p in range(4):
            if not all_preds[m_col][p]: continue
            combined = pd.concat(all_preds[m_col][p]).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            ax.plot(combined.index, combined.values, color=colors[p], linestyle='--', 
                    marker='o', markersize=4, alpha=0.8, label=f"予測フェーズ{p+1}", zorder=3)

        ax.set_title(f"実績 vs 当てはめ vs 予測 : {orig_name}", fontsize=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.2)
        
        plt.tight_layout()
        save_path = output_root / f"全期間予測比較_{orig_name}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"✅ グラフ生成完了: {save_path}")


def plot_exog_trends(csv_path, output_dir, target_var="GDP"):
    output_root = Path(output_dir)
    df = pd.read_csv(csv_path)

    sub_df = df[(df["Target_Variable"] == target_var) & (df["Model_Type"].str.contains("PC"))].copy()
    if sub_df.empty:
        return

    sub_df["EndDateStr"] = sub_df["Window"].str.split("_").str[-1]
    sub_df["Date"] = pd.to_datetime(sub_df["EndDateStr"], format="%Y%m%d")
    sub_df = sub_df.sort_values("Date")

    models = sub_df["Model_Type"].unique()
    pc_cols = [c for c in sub_df.columns if c.startswith("PC")]

    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, m_type in zip(axes, models):
        m_data = sub_df[sub_df["Model_Type"] == m_type]
        active_pcs = [c for c in pc_cols if not m_data[c].isna().all()]

        for col in active_pcs:
            ax.plot(m_data["Date"], m_data[col], marker="o", label=col, lw=2)

        ax.set_title(f"Target: {target_var} | Model: {m_type}", fontsize=12, fontweight="bold")
        ax.axhline(0.3, color="red", ls="--", alpha=0.6, label="LB: 0.3")
        ax.axhline(0, color="black", lw=1, alpha=0.3)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))

        ax.set_ylabel("Coefficient")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="both", alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = output_root / f"係数推移_{target_var}_日付修正版.png"
    plt.savefig(save_path, dpi=200, facecolor="white")
    plt.close()
    print(f"✅ 日付を修正して保存しました: {save_path}")


def build_var_matrices(Beta, m_dim, p_lag):
    """
    Beta: (k, m_dim) where k = 1 + m_dim*p + exog_dim
    戻り: A_list (len=p), companion_F (m_dim*p, m_dim*p)
    """
    # lag係数の取り出し（Xの並びは [const, lag1(m), lag2(m), ... lagp(m), exog...]
    A_list = []
    for lag in range(p_lag):
        start = 1 + lag * m_dim
        end = start + m_dim
        A_list.append(Beta[start:end, :].T)  # (m_dim, m_dim)

    if p_lag == 1:
        F = A_list[0]
        return A_list, F

    # companion matrix
    F = np.zeros((m_dim * p_lag, m_dim * p_lag))
    # first block row: [A1 A2 ... Ap]
    F[:m_dim, : m_dim * p_lag] = np.hstack(A_list)
    # subdiagonal identity
    F[m_dim:, :-m_dim] = np.eye(m_dim * (p_lag - 1))
    return A_list, F


def dynamic_forecast_scaled(Beta, X_te, te_s_exog, m_dim, p_lag, steps):
    """
    逐次予測（スケール済み）
    X_te: design matrix on test (valid rows)
    te_s_exog: dataframe (test exog) aligned with test dates (valid rows)  ※comboが空なら None
    """
    if X_te is None or len(X_te) == 0:
        return None

    # 初期ラグはX_te[0]から取得
    curr = X_te[0:1, :].copy()
    y_preds = []

    for t in range(steps):
        yhat = curr @ Beta  # (1, m_dim)
        y_preds.append(yhat)

        if t == steps - 1:
            break

        # lagブロック更新（p_lag対応）
        lag_block = curr[0, 1 : 1 + m_dim * p_lag].reshape(p_lag, m_dim)  # [lag1; lag2; ...]
        lag_block = np.vstack([yhat.reshape(1, m_dim), lag_block[:-1, :]])  # 新lag1=予測、古いのを後ろへ

        # 次の外生（あれば実績）
        if te_s_exog is not None and not te_s_exog.empty:
            if (t + 1) < len(te_s_exog):
                next_exog = te_s_exog.iloc[t + 1 : t + 2].values
            else:
                next_exog = te_s_exog.iloc[-1:].values
        else:
            next_exog = np.empty((1, 0))

        curr = np.concatenate([[[1.0]], lag_block.reshape(1, -1), next_exog], axis=1)

    return np.vstack(y_preds)


def get_bound(STRICT_CONSTRAINTS, orig_exog, orig_target):
    """
    優先順位：
    1) (orig_exog, orig_target)
    2) ("*", orig_target)
    3) (orig_exog, "*")
    4) ("*", "*")
    5) さらに「target側が部分一致」も許す（例: ("*", "UNEMP") が UNEMP_RATE に効く）
    """
    b = (
        STRICT_CONSTRAINTS.get((orig_exog, orig_target))
        or STRICT_CONSTRAINTS.get(("*", orig_target))
        or STRICT_CONSTRAINTS.get((orig_exog, "*"))
        or STRICT_CONSTRAINTS.get(("*", "*"))
    )
    if b is not None:
        return b

    # 部分一致（target）
    for (ex, tg), val in STRICT_CONSTRAINTS.items():
        if ex not in [orig_exog, "*"]:
            continue
        if tg == "*":
            continue
        if tg in orig_target:
            return val

    return None


# =========================================================
# 3. メイン処理
# =========================================================
def main():
    df_raw = pd.read_csv(CONFIG["data_path"], index_col=0, parse_dates=True).sort_index()
    sector_df, macro_df, common_idx, meta = prepare_aligned(df_raw)

    test_len = CONFIG["test_steps"]
    n_total = len(common_idx)
    all_coefficients_records = []

    for i in range(n_total - test_len + 1):
        te_idx_list = np.arange(i, i + test_len)
        tr_idx_list = np.setdiff1d(np.arange(n_total), te_idx_list)

        d_start = common_idx[te_idx_list[0]]
        d_end = common_idx[te_idx_list[-1]]

        root_dir = Path(CONFIG["output_dir"]) / f"Window_Test_{d_start:%Y%m%d}_{d_end:%Y%m%d}"
        root_dir.mkdir(parents=True, exist_ok=True)

        print(f"▶️ 実行中: {root_dir.name}")

        # ===== PCA Fit (train=テスト抜いた全期間) =====
        sc_pca = StandardScaler()
        X_tr_pca = sc_pca.fit_transform(sector_df.iloc[tr_idx_list])
        pca_temp = PCA().fit(X_tr_pca)
        n_pcs = max(3, np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= 0.90) + 1)

        pca = PCA(n_components=n_pcs).fit(X_tr_pca)
        pc_cols_orig = [f"PC{k+1}" for k in range(n_pcs)]

        # ★符号固定ルール（ここで一元管理）
        sector_groups = {
            "PC1": SECTOR_COLS,
            "PC2": ["RET_FOODS", "RET_RETAIL_TRADE", "RET_PHARMACEUTICAL"],
            "PC3": ["RET_MACHINERY", "RET_ELEC_APPLIANCES_PRECISION", "RET_STEEL_NONFERROUS"],
        }
        fixed_signs = decide_pc_signs_by_groups(pca, sector_df, sector_groups, top_n_for_sign=5, n_fix=3)

        # ===== 全期間PCスコア（符号固定→diff→モデル投入） =====
        full_pcs = pca.transform(sc_pca.transform(sector_df))
        full_pcs[:, :3] *= fixed_signs  # ←ここが「モデル投入が逆？」の核心修正

        pc_diff_all = pd.DataFrame(full_pcs, index=sector_df.index).diff().dropna()
        pc_diff_all = pc_diff_all.iloc[:, :3]
        pc_diff_all.columns = [f"PC{k+1}_DIFF" for k in range(3)]

        # ===== レポートも同じ符号系で統一 =====
        analyze_pca_details(
            pca,
            sector_df,
            pc_cols_orig,
            root_dir,
            tr_idx_list[0],
            tr_idx_list[-1],
            test_len,
            fixed_signs=fixed_signs,
        )

        # マクロ同期
        m_df_win = macro_df.loc[pc_diff_all.index]

        pc_cols_diff = list(pc_diff_all.columns)
        combinations = [[]] + [list(c) for r in range(1, 4) for c in itertools.combinations(pc_cols_diff, r)]
        summary_list = []

        for combo in combinations:
            m_name_display = ", ".join(combo) if combo else "マクロのみ"
            sub_dir = root_dir / ("_".join(combo) if combo else "BASE_VAR_ONLY")
            sub_dir.mkdir(parents=True, exist_ok=True)

            win_idx = m_df_win.index
            tr_pre_dates = common_idx[tr_idx_list[tr_idx_list < te_idx_list[0]]]
            tr_post_dates = common_idx[tr_idx_list[tr_idx_list > te_idx_list[-1]]]
            te_dates = common_idx[te_idx_list]

            # --- 表示レンジを先に確保（fitted作成でも使う） ---
            start_d_idx = np.where(common_idx == te_dates[0])[0][0]
            plot_start_idx = max(0, start_d_idx - 8)
            plot_end_idx = min(len(common_idx), start_d_idx + test_len + 4)
            full_display_range = common_idx[plot_start_idx:plot_end_idx]

            tr_pre = pd.concat(
                [m_df_win.loc[win_idx.isin(tr_pre_dates)], pc_diff_all.loc[win_idx.isin(tr_pre_dates), combo]],
                axis=1,
            )
            tr_post = pd.concat(
                [m_df_win.loc[win_idx.isin(tr_post_dates)], pc_diff_all.loc[win_idx.isin(tr_post_dates), combo]],
                axis=1,
            )

            # 未来側の先頭をNaN（ラグ接続防止）
            if not tr_post.empty:
                tr_post.iloc[0, :] = np.nan

            tr_raw = pd.concat([tr_pre, tr_post])
            te_raw = pd.concat(
                [m_df_win.loc[win_idx.isin(te_dates)], pc_diff_all.loc[win_idx.isin(te_dates), combo]],
                axis=1,
            )

            means = tr_raw.mean()
            stds = tr_raw.std(ddof=0).replace(0, 1.0)
            tr_s, te_s = (tr_raw - means) / stds, (te_raw - means) / stds

            m_cols = list(macro_df.columns)
            m_dim = len(m_cols)
            p_lag = CONFIG["p_lag"]

            Y_tr, X_tr, idx_tr = make_design(tr_s[m_cols], tr_s[combo] if combo else None, CONFIG["p_lag"])

            # ===== 制約 + Ridge（lsq_linear） =====
            STRICT_CONSTRAINTS = {
                ("PC1", "GDP"): (0.3, 0.7),
                ("PC1", "NIKKEI"): (0.3, 0.7),
                ("PC1", "CPI"): (0.1, 0.5),
                ("PC1", "UNEMP_RATE"): (-0.7, -0.3),
                ("*", "UNEMP"): (-np.inf, -0.3),  # UNEMP_RATE に部分一致で効かせる
            }

            # param名（CSV用）
            param_names = (
                ["CONST"]
                + [f"LAG{lag}_{m}" for lag in range(1, p_lag + 1) for m in m_cols]
                + list(combo)
            )

            # Ridge augmentation
            ridge_aug = np.sqrt(CONFIG["ridge"]) * np.eye(X_tr.shape[1])
            X_aug = np.vstack([X_tr, ridge_aug])

            beta_list = []
            for j, target_m_col in enumerate(m_cols):
                orig_target = meta[target_m_col]["orig"]
                Y_aug = np.concatenate([Y_tr[:, j], np.zeros(X_tr.shape[1])])

                lb = np.full(X_aug.shape[1], -np.inf)
                ub = np.full(X_aug.shape[1], np.inf)

                # exog係数の開始位置（p_lag対応）
                start_exog_idx = 1 + m_dim * p_lag

                if CONFIG["use_constraints"] and combo:
                    for k, exog_pc_name in enumerate(combo):
                        orig_exog = exog_pc_name.split("_")[0]  # "PC1" など
                        target_idx = start_exog_idx + k

                        bound = get_bound(STRICT_CONSTRAINTS, orig_exog, orig_target)
                        if bound is not None:
                            lb[target_idx] = bound[0]
                            ub[target_idx] = bound[1]

                if CONFIG["use_constraints"]:
                    res = lsq_linear(X_aug, Y_aug, bounds=(lb, ub), lsmr_tol="auto")
                    beta = res.x
                else:
                    # 制約なしの場合もridge付きの最小二乗
                    beta = np.linalg.lstsq(X_aug, Y_aug, rcond=None)[0]

                beta_list.append(beta)

                # 係数保存
                record = {"Window": root_dir.name, "Model_Type": m_name_display, "Target_Variable": orig_target}
                for name, val in zip(param_names, beta):
                    record[name] = val
                all_coefficients_records.append(record)

            Beta = np.array(beta_list).T  # (k, m_dim)

            # ================================
            # 当てはめ値（1-step）: “全日”で算出（test window以外も含む）
            # 置き場所：Beta = ... の直後、tr_resid の直前
            # ================================

            # 1) “全日”の説明変数（マクロ変化量 + PC）を作る（win_idx = m_df_win.index）
            full_raw = pd.concat(
                [
                    m_df_win,  # マクロ（変化量）
                    pc_diff_all.loc[m_df_win.index, combo] if combo else pd.DataFrame(index=m_df_win.index),
                ],
                axis=1,
            )

            # 2) 学習で使った mean/std（tr_raw由来）で全日をスケール
            full_s = (full_raw - means) / stds

            # 3) 全日デザイン行列（lagは “全日” の実績変化量から作る）
            Y_all, X_all, idx_all = make_design(
                full_s[m_cols],
                full_s[combo] if combo else None,
                p_lag,
            )

            # 4) 1-stepの変化量予測（スケール済み） → スケール解除
            Yhat_all_scaled = X_all @ Beta
            fit_change_df = pd.DataFrame(Yhat_all_scaled, index=idx_all, columns=m_cols)
            fit_change_unscaled = fit_change_df.mul(stds[m_cols], axis=1).add(means[m_cols], axis=1)

            # 5) 変化量 → 水準（t-1実績がある日だけ）
            fit_level_map = {}

            for m_col in m_cols:
                orig_name = meta[m_col]["orig"]
                meth = meta[m_col]["method"]

                dc = fit_change_unscaled[m_col]  # 予測された変化量（LOGDIFF/DIFF）
                prev_level = df_raw[orig_name].shift(1).reindex(idx_all)  # t-1の実績水準

                if meth == "LOGDIFF":
                    fitted_level = prev_level * np.exp(dc)
                else:
                    fitted_level = prev_level + dc

                fitted_level = fitted_level.rename(f"{orig_name}_Fitted")
                fit_level_map[orig_name] = fitted_level

                out_fit = pd.concat(
                    [
                        df_raw[orig_name].reindex(idx_all).rename(f"{orig_name}_Actual"),
                        fitted_level,
                        dc.rename(f"{orig_name}_FittedChange"),
                    ],
                    axis=1,
                )
                out_fit.to_csv(sub_dir / f"当てはめ_{orig_name}.csv", encoding="utf-8-sig")
            # ===== 指標 =====
            tr_resid = Y_tr - X_tr @ Beta
            n_obs = len(Y_tr)
            tr_rmse = np.sqrt(np.mean(tr_resid**2))

            Y_te, X_te, idx_te = make_design(te_s[m_cols], te_s[combo] if combo else None, CONFIG["p_lag"])
            te_rmse = np.sqrt(np.mean((Y_te - X_te @ Beta) ** 2)) if len(Y_te) > 0 else np.nan

            sigma_matrix = (tr_resid.T @ tr_resid) / max(n_obs, 1)
            sign, logdet = np.linalg.slogdet(sigma_matrix)
            if sign > 0:
                aic_val = n_obs * logdet + 2 * Beta.size
            else:
                eigs = np.linalg.eigvalsh(sigma_matrix)
                valid_eigs = eigs[eigs > 1e-15]
                aic_val = n_obs * np.sum(np.log(valid_eigs)) + 2 * Beta.size if len(valid_eigs) > 0 else np.nan

            # VAR安定性（p_lag対応：companionの最大固有値）
            A_list, F = build_var_matrices(Beta, m_dim, p_lag)
            max_eig_abs = np.max(np.abs(np.linalg.eigvals(F))) if p_lag > 1 else np.max(np.abs(np.linalg.eigvals(A_list[0])))

            summary_list.append(
                {
                    "モデル構成": m_name_display,
                    "訓練RMSE": round(tr_rmse, 4),
                    "予測RMSE": round(te_rmse, 4) if not np.isnan(te_rmse) else np.nan,
                    "RMSE比": round(te_rmse / tr_rmse, 2) if tr_rmse > 0 and not np.isnan(te_rmse) else np.nan,
                    "AIC": round(aic_val, 2) if not np.isnan(aic_val) else np.nan,
                    "最大固有値": round(float(max_eig_abs), 3),
                }
            )

            # ===== 逐次予測（p_lag対応）→ 水準復元→保存＆図 =====
            if len(X_te) > 0:
                te_exog = te_s[combo] if (combo and combo is not None) else None
                y_te_pred_scaled = dynamic_forecast_scaled(Beta, X_te, te_exog, m_dim, p_lag, CONFIG["test_steps"])
            else:
                y_te_pred_scaled = None

            if y_te_pred_scaled is not None and i != 0:
                start_d_idx = np.where(common_idx == te_dates[0])[0][0]
                te_actual_dates = common_idx[start_d_idx : start_d_idx + CONFIG["test_steps"]]

                plot_start_idx = max(0, start_d_idx - 8)
                plot_end_idx = min(len(common_idx), start_d_idx + CONFIG["test_steps"] + 4)
                full_display_range = common_idx[plot_start_idx:plot_end_idx]

                for i_m, m_col in enumerate(m_cols):
                    orig_name = meta[m_col]["orig"]
                    meth = meta[m_col]["method"]

                    pred_change = (y_te_pred_scaled[:, i_m] * stds[m_col]) + means[m_col]
                    # 変化の基準になる「直前水準」（あなたの元実装踏襲）
                    last_actual_level = df_raw.loc[:te_actual_dates[0], orig_name].iloc[-2]

                    pred_levels = []
                    curr_level = last_actual_level
                    for val in pred_change:
                        if meth == "LOGDIFF":
                            curr_level = curr_level * np.exp(val)
                        else:
                            curr_level = curr_level + val
                        pred_levels.append(curr_level)

                    p_df = pd.DataFrame(pred_levels, index=te_actual_dates, columns=[f"{orig_name}_Pred"])
                    p_df.to_csv(sub_dir / f"予測値_水準ベース_{orig_name}.csv", encoding="utf-8-sig")

                    plt.figure(figsize=(11, 5.5))
                    plt.plot(
                        full_display_range,
                        df_raw.loc[full_display_range, orig_name],
                        label="実績値",
                        color="#333333",
                        marker="o",
                        markersize=5,
                        alpha=0.8,
                    )
                    plt.plot(
                        te_actual_dates,
                        pred_levels,
                        label="予測値 (逐次)",
                        color="red",
                        linestyle="--",
                        marker="x",
                        markersize=7,
                        lw=2,
                    )
                    # --- 当てはめ値（学習期間の1-step fitted）を重ねる ---
                    fit_csv = sub_dir / f"当てはめ_{orig_name}.csv"
                    # --- 当てはめ値（1-step fitted）を重ねる（その場で作ったfit_level_mapから） ---
                    if orig_name in fit_level_map:
                        fitted_part = fit_level_map[orig_name].reindex(full_display_range)

                        # テスト期間は当てはめ線を消す（見た目きれい）
                        fitted_part.loc[te_actual_dates] = np.nan

                        plt.plot(
                            fitted_part.index,
                            fitted_part.values,
                            linestyle="--",
                            marker=".",
                            markersize=4,
                            lw=1.5,
                            label="当てはめ値(1-step)",
                        )

                    plt.axvspan(te_actual_dates[0], te_actual_dates[-1], color="gray", alpha=0.1, label="予測対象期間")

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[3, 6, 9, 12]))
                    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y年%m月"))

                    plt.title(f"【予測】{orig_name} : {m_name_display}", fontsize=14)
                    plt.xlabel("年月", fontsize=12)
                    plt.ylabel("水準 (Level)", fontsize=12)
                    plt.legend(loc="best", frameon=True, shadow=True)
                    plt.grid(True, which="major", linestyle="--", alpha=0.4)
                    plt.tight_layout()

                    plt.savefig(sub_dir / f"PRED_{orig_name}.png", dpi=150)
                    plt.close()

            # ===== IRF（p_lag対応：companionで伝播） =====
            if CONFIG["do_irf"] and combo:
                # exog係数 B の取り出し（p_lag対応）
                exog_start = 1 + m_dim * p_lag
                Bmat = Beta[exog_start:, :].T  # (m_dim, exog_dim)

                for j, pc_label in enumerate(combo):
                    pc_folder = sub_dir / f"Shock_{pc_label}"
                    pc_folder.mkdir(parents=True, exist_ok=True)

                    # 状態ベクトル: [y_t, y_{t-1}, ..., y_{t-p+1}]  (m_dim*p)
                    state_dim = m_dim * p_lag
                    impact_state = np.zeros((13, state_dim))

                    # t=1 で yにショック（元実装に合わせて 1期目に置く）
                    shock_y = np.zeros(m_dim)
                    shock_y[:] = Bmat[:, j]
                    impact_state[1, :m_dim] = shock_y

                    # 伝播
                    for h in range(2, 13):
                        if p_lag == 1:
                            impact_state[h, :m_dim] = A_list[0] @ impact_state[h - 1, :m_dim]
                        else:
                            impact_state[h] = F @ impact_state[h - 1]

                    # y成分だけ取り出して「水準」へ復元＆図
                    for i_m, m_col in enumerate(m_cols):
                        orig_m = meta[m_col]["orig"]
                        meth_m = meta[m_col]["method"]
                        base = BASE_LEVELS.get(orig_m, 100)

                        # 変化量（スケール戻し）
                        imp_raw = impact_state[:, i_m] * stds[m_col]  # y成分だけ
                        vals = [base]
                        for s in range(1, 13):
                            if meth_m == "LOGDIFF":
                                vals.append(vals[-1] * np.exp(imp_raw[s]))
                            else:
                                vals.append(vals[-1] + imp_raw[s])

                        plt.figure(figsize=(6, 3.5))
                        ax = plt.gca()
                        ax.yaxis.get_major_formatter().set_useOffset(False)
                        ax.yaxis.get_major_formatter().set_scientific(False)

                        plt.plot(range(13), vals, color="k", lw=1)
                        plt.scatter(range(13), vals, marker="o", s=15, color="k", zorder=2)
                        plt.scatter(1, vals[1], facecolors="none", edgecolors="k", marker="o", s=100, lw=1, zorder=3)
                        plt.axvline(x=1, color="k", ls="--", lw=0.7, alpha=0.6)
                        plt.axhline(base, color="gray", ls=":", lw=0.8)
                        plt.title(f"{pc_label} → {orig_m}")
                        plt.grid(alpha=0.15)
                        plt.tight_layout()
                        plt.savefig(pc_folder / f"{orig_m}_応答.png")
                        plt.close()

        # ウィンドウごとのサマリー保存
        pd.DataFrame(summary_list).sort_values("AIC").to_csv(
            root_dir / "モデル比較サマリー.csv", index=False, encoding="utf-8-sig"
        )

    # ===== 全体集計 =====
    aggregate_results(CONFIG["output_dir"])

    output_folder = Path(CONFIG["output_dir"])
    csv_file = output_folder / "全ウィンドウ集計_モデル評価_詳細版.csv"
    if csv_file.exists():
        visualize_model_performance(csv_file)

    print("📊 全期間の連結予測グラフを生成中...")
    plot_connected_forecasts(CONFIG["output_dir"], df_raw, meta)

    if all_coefficients_records:
        df_coef = pd.DataFrame(all_coefficients_records)
        output_path = Path(CONFIG["output_dir"]) / "all_model_coefficients.csv"
        df_coef.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 全ウィンドウの係数詳細を保存しました: {output_path}")

    print(f"✅ 全工程完了。{CONFIG['output_dir']} 内の画像とCSVを確認してください。")

    output_path = Path(CONFIG["output_dir"]) / "all_model_coefficients.csv"
    for t in ["GDP", "NIKKEI", "USD_JPY"]:
        plot_exog_trends(output_path, CONFIG["output_dir"], target_var=t)


if __name__ == "__main__":
    main()