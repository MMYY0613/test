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
from matplotlib import patheffects as pe

warnings.simplefilter('ignore')

# =========================================================
# 1. 設定・定数
# =========================================================
CONFIG = {
    "data_path": "./data/all_q_merged_new_tmp.csv",
    "output_dir": "./output_mw_new_tmp_3",
    "test_steps": 4,
    "pc_max": 3,
    "p_lag": 1,
    "ridge": 1.0,
    "do_irf": True,
    "verbose": False,
}

BASE_LEVELS = {
    "GDP": 500000, "NIKKEI": 20000, "USD_JPY": 150, "UNEMP_RATE": 3.0,
    "JGB_1Y": 0.0, "JGB_3Y": 0.0, "JGB_10Y": 0.0, "CPI": 0.0, "TOPIX": 1500,
}

TARGET_MACRO = list(BASE_LEVELS.keys())

SECTOR_COLS = [
    "ガラス・土石製品",
    "ゴム製品",
    "電気機器",
    "金属製品",
    "その他製品",
    "機械",
    "食料品",
    "輸送用機器",
    "化学",
    "電気・ガス業",
    "鉱業",
    "鉄鋼",
    "精密機器",
    "石油・石炭製品",
]

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = ["Hiragino Sans"]

# =========================================================
# 2. ロジック関数
# =========================================================
def log(*args, **kwargs):
    if CONFIG.get("verbose", True):
        print(*args, **kwargs)

def smart_transform(series, name):
    # CPIが前年比(%)なら “そのまま” 使う（変換しない）
    if name == "CPI":
        return series, "LEVEL"

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
    s_raw = df_raw[[c for c in SECTOR_COLS if c in df_raw.columns]].interpolate(limit_direction='both')
    s_diff = s_raw.diff().dropna()
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
    return Y[valid], X[valid]

def analyze_pca_details(pca, scaler, sector_df, pc_cols, root_dir, tr_idx_list, te_idx_list):
    pca_dir = root_dir / "pca_analysis"
    pca_dir.mkdir(exist_ok=True)

    idx = np.r_[tr_idx_list, te_idx_list]
    X_target = sector_df.iloc[idx].sort_index()
    X_scaled = scaler.transform(X_target)
    scores = pca.transform(X_scaled)
    score_df = pd.DataFrame(scores, index=X_target.index, columns=pc_cols)

    expl = pca.explained_variance_ratio_
    components = pca.components_.copy()

    score_df.to_csv(pca_dir / "主成分スコア_生データ.csv", encoding="utf-8-sig")

    loadings = pd.DataFrame(components.T, index=sector_df.columns, columns=pc_cols)
    loadings.to_csv(pca_dir / "セクターの負荷量_一覧.csv", encoding="utf-8-sig")

    expl_df = pd.DataFrame(expl.reshape(1, -1), columns=pc_cols, index=["ExplainedVariance"])
    expl_df.to_csv(pca_dir / "主成分_寄与率.csv", encoding="utf-8-sig")

    # スコア推移プロット（PC3まで）
    plt.figure(figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, pc in enumerate(pc_cols[:3]):
        plt.plot(score_df.index, score_df[pc], label=f"{pc} (寄与:{expl[i]*100:.1f}%)", color=colors[i], lw=2)
    plt.title("主成分スコアの推移 (セクター標準化後の生値)")

    # --- 軸ラベル追加 ---
    plt.xlabel("日付")
    plt.ylabel("主成分スコア")

    plt.axhline(0, color='black', lw=1); plt.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(pca_dir / "PCA_スコア推移_生値.png"); plt.close()

    # 負荷量の棒グラフ（PC3まで）
    for i, pc in enumerate(pc_cols[:3]):
        plt.figure(figsize=(10, 5))
        loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(10).plot(kind='bar', color=colors[i])
        plt.title(f"{pc} セクターの負荷量 (寄与:{expl[i]*100:.1f}%)")

        # --- 軸ラベル追加 ---
        plt.xlabel("セクター")
        plt.ylabel("負荷量")

        plt.axhline(0, color='black', lw=1); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig(pca_dir / f"負荷量_{pc}_TOP10.png"); plt.close()

    print(f"✅ PCA分析完了: {pca_dir}")

def plot_abs_heatmaps(full_pca_df, output_root):
    # PCごとにループして可視化
    for pc in ["PC1", "PC2", "PC3"]:
        target_df = full_pca_df[full_pca_df["PC_Type"] == pc]
        if target_df.empty: continue

        # ウィンドウ名をインデックスにして、セクター列のみ抽出（数値データのみ）
        # 文字列カラムを除外して転置
        plot_data = target_df.drop(columns=["PC_Type", "Window", "Explained_Variance"]).T
        plot_data = plot_data.abs()
        plot_data.columns = target_df["Window"] # x軸をウィンドウ名に

        plt.figure(figsize=(14, 9))

        # 負荷量の絶対値なので、0からの強弱がはっきりする "Reds" などを採用
        # vmin=0, vmax=0.5 などでスケールを固定するとウィンドウ間の比較がしやすくなります
        sns.heatmap(plot_data,
                    cmap="Reds",
                    annot=False,
                    cbar_kws={'label': f'{pc} 負荷量（絶対値）'},
                    vmin=0,
                    vmax=max(0.5, plot_data.max().max()))

        # 寄与率を取得
        mean_var = target_df["Explained_Variance"].mean() * 100
        plt.title(f"{pc} 構成セクターの負荷量推移（絶対値）\n[平均寄与率: {mean_var:.1f}%]", fontsize=14)
        plt.xlabel("テストウィンドウ (Window)", fontsize=12)
        plt.ylabel("セクター", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_root / f"全期間_{pc}_構造推移_絶対値.png")
        plt.close()

def plot_signed_heatmaps(full_pca_df, output_root):
    for pc in ["PC1", "PC2", "PC3"]:
        target_df = full_pca_df[full_pca_df["PC_Type"] == pc]
        if target_df.empty:
            continue

        plot_data = target_df.drop(columns=["PC_Type", "Window", "Explained_Variance"]).T
        plot_data.columns = target_df["Window"]

        vmax = np.nanmax(np.abs(plot_data.values))
        vmax = max(0.5, float(vmax))  # 見やすさの下限

        plt.figure(figsize=(14, 9))
        sns.heatmap(
            plot_data,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=False,
            cbar_kws={"label": f"{pc} 負荷量（符号つき）"},
        )

        mean_var = target_df["Explained_Variance"].mean() * 100
        plt.title(f"{pc} 構成セクターの負荷量推移（符号つき）\n[平均寄与率: {mean_var:.1f}%]", fontsize=14)
        plt.xlabel("テストウィンドウ (Window)")
        plt.ylabel("セクター")
        plt.tight_layout()
        plt.savefig(output_root / f"全期間_{pc}_構造推移_符号つき.png")
        plt.close()

def aggregate_results(output_root, agg_root=None):
    output_root = Path(output_root)
    agg_root = Path(agg_root) if agg_root is not None else output_root
    all_summaries = []
    pca_abs_details = []

    # 1. 各ウィンドウを巡回して生データを収集
    for window_path in sorted(output_root.glob("Window_Test_*")):
        window_name = window_path.name.replace("Window_Test_", "")

        # モデル比較サマリー
        summary_file = window_path / "モデル比較サマリー.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df["Test_Window"] = window_name
            all_summaries.append(df)

        # PCA負荷量（絶対値）
        loadings_file = window_path / "pca_analysis" / "セクターの負荷量_一覧.csv"
        variance_file = window_path / "pca_analysis" / "主成分_寄与率.csv"
        if loadings_file.exists() and variance_file.exists():
            ld_df = pd.read_csv(loadings_file, index_col=0)
            vr_df = pd.read_csv(variance_file, index_col=0)
            for pc in ["PC1", "PC2", "PC3"]:
                if pc in ld_df.columns:
                    row = ld_df[pc].to_frame().T   # ← abs() を消す
                    row.index = [f"{window_name}_{pc}"]
                    row.insert(0, "Window", window_name)
                    row.insert(1, "PC_Type", pc)
                    row.insert(2, "Explained_Variance", vr_df.at["ExplainedVariance", pc])
                    pca_abs_details.append(row)

    if not all_summaries:
        return

    # --- 2. モデル評価の詳細統計（忘れてないポイント：0除外 & 日本語カラム） ---
    # ignore_index=True を追加して、インデックスの重複を解消する
    merged_summary = pd.concat(all_summaries, ignore_index=True)

    plot_final_boxplots(merged_summary, agg_root / "model_eval")

    target_cols = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]
    stats_list = []
    for col in target_cols:
        if col in merged_summary.columns:
            temp_series = merged_summary.copy()
            if col != "AIC": # AIC以外は0を外す
                temp_series[col] = temp_series[col].replace(0, np.nan)

            res = temp_series.groupby("モデル構成")[col].agg([
                (f"{col}_平均", "mean"),
                (f"{col}_標準偏差", "std"),
                (f"{col}_最大", "max"),
                (f"{col}_最小", "min")
            ])
            stats_list.append(res)

    model_perf_detail = pd.concat(stats_list, axis=1)
    model_perf_detail["有効試行回数"] = merged_summary.groupby("モデル構成")["予測RMSE"].apply(lambda x: (x != 0).sum())
    model_perf_detail = model_perf_detail.round(4)
    model_perf_detail.to_csv(agg_root / "model_eval" / "全ウィンドウ集計_モデル評価_詳細版.csv", encoding="utf-8-sig")

    # --- 3. PCA構造の集計とヒートマップ（ここも忘れてません！） ---
    if pca_abs_details:
        full_pca_df = pd.concat(pca_abs_details)
        full_pca_df.to_csv(agg_root / "pca" / "全ウィンドウ集計_PCA構造_符号つき詳細.csv", encoding="utf-8-sig")
        # 絶対値ヒートマップを保存
        plot_signed_heatmaps(full_pca_df, agg_root / "heatmaps")
        plot_abs_heatmaps(full_pca_df, agg_root / "heatmaps")
    print(f"✅ すべての集計・画像保存（ヒートマップ、箱ひげ図、統計詳細）が完了しました。")

def plot_final_boxplots(merged_summary, output_root):
    """
    あなたが『ええ感じ』と言った、色付きの箱ひげ図 ＋ 統計量
    """
    output_root = Path(output_root)
    plot_df_base = merged_summary.copy().reset_index(drop=True)

    targets = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]

    for col in targets:
        if col not in plot_df_base.columns: continue

        plt.figure(figsize=(14, 8))

        # 0除外（AIC以外）
        plot_data = plot_df_base[plot_df_base[col] != 0].copy() if col != "AIC" else plot_df_base.copy()
        if plot_data.empty: continue

        # 平均値が低い順にソートして並べる
        order = plot_data.groupby("モデル構成")[col].mean().sort_values().index

        # 1. 【メイン】色付きの箱ひげ図（これが『ええ感じ』の正体）
        sns.boxplot(data=plot_data, x="モデル構成", y=col, order=order,
                    palette="Set3", width=0.6, boxprops=dict(alpha=0.7))

        # 2. 平均値(◆)と標準偏差(赤線)を重ねる
        sns.pointplot(data=plot_data, x="モデル構成", y=col, order=order,
                      join=False, color="red", marker="D", scale=0.7,
                      errorbar="sd", capsize=.15, label="平均 ± 標準偏差")

        # 3. 生データ（薄い点）はノイズになるので、ここでは除外か極薄に
        # sns.stripplot(data=plot_data, x="モデル構成", y=col, order=order, color="black", alpha=0.1, jitter=True)

        plt.xticks(rotation=45, ha='right')
        plt.title(f"{col}の箱ひげ図", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend(loc='upper left')
        plt.tight_layout()

        plt.savefig(output_root / f"全期間_箱ひげ_{col}.png", dpi=300)
        plt.close()

def visualize_model_performance(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    output_dir = Path(csv_path).parent
    plt.style.use('ggplot')

    # --- AIC vs 予測RMSE の散布図だけ作る ---
    if "AIC_平均" in df.columns and "予測RMSE_平均" in df.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df,
            x="AIC_平均",
            y="予測RMSE_平均",
            size="予測RMSE_標準偏差" if "予測RMSE_標準偏差" in df.columns else None,
            hue=df.index,
            sizes=(100, 1000),
            alpha=0.6
        )

        for i, txt in enumerate(df.index):
            plt.annotate(
                txt,
                (df["AIC_平均"].iloc[i], df["予測RMSE_平均"].iloc[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        plt.title("モデルの複雑さ(AIC) vs 予測精度(RMSE)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "全期間_AIC_vs_RMSE_分布.png", dpi=300)
        plt.close()

    print(f"💾 散布図を {output_dir} に保存しました。")

def contiguous_train_span(n_total, te_idx_list):
    """
    analyze_pca_details が「連続区間」を前提にしてるので、
    trainの前半/後半のうち長い方の連続区間 [t_start, t_end) を返す。
    """
    te_start = int(te_idx_list[0])
    te_end = int(te_idx_list[-1])

    pre_len = te_start
    post_len = n_total - (te_end + 1)

    if pre_len >= post_len:
        return 0, te_start
    else:
        return te_end + 1, n_total

SECTOR_GROUPS = {
    "PC1": SECTOR_COLS,
    "PC2": ["石油・石炭製品", "鉱業", "電気・ガス業"],
    "PC3": ["食料品"],
}

def fix_pca_sign_inplace(pca, feature_names, pc_cols, sector_groups, top_n=5):
    for pc_name in pc_cols:
        if pc_name not in sector_groups:
            continue

        # PC1 -> 0, PC2 -> 1 ...
        i = int(pc_name.replace("PC", "")) - 1

        target_sectors = sector_groups[pc_name]
        load = pd.Series(pca.components_[i], index=feature_names)

        group_load = load[load.index.isin(target_sectors)]
        if group_load.empty:
            continue

        top_idx = group_load.abs().sort_values(ascending=False).head(top_n).index
        sign_check = group_load.loc[top_idx].mean()

        if sign_check < 0:
            pca.components_[i] *= -1

from datetime import datetime

def _parse_window_start_date(window_path: Path):
    # Window_Test_20080630_20090331 -> 20080630
    parts = window_path.name.split("_")
    start_str = parts[2]
    return pd.to_datetime(start_str, format="%Y%m%d")

def _quarter_index(dt: pd.Timestamp):
    # 四半期末想定（3,6,9,12月）。念のため一般化。
    q = (dt.month - 1) // 3 + 1
    return dt.year * 4 + (q - 1)

def plot_connected_forecasts(window_root, save_root, df_raw, meta, model_subdir="PC1_DIFF", last_years=None):
    plt.style.use('default')
    window_root = Path(window_root)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # ★CSVだけ1段深い場所へ
    csv_root = save_root / "csv" / model_subdir
    csv_root.mkdir(parents=True, exist_ok=True)

    m_cols_transformed = list(meta.keys())

    window_dirs = [p for p in window_root.glob("Window_Test_*") if p.is_dir()]
    window_dirs = sorted(window_dirs, key=_parse_window_start_date)
    if not window_dirs:
        print("Window_Test_* が見つかりません")
        return

    start0 = _parse_window_start_date(window_dirs[0])
    q0 = _quarter_index(start0)

    # phaseごとに予測を格納
    all_preds = {m: {p: [] for p in range(4)} for m in m_cols_transformed}
    rmse_list = {m: {p: [] for p in range(4)} for m in m_cols_transformed}

    for window_path in window_dirs:
        sub = window_path / model_subdir
        if not sub.exists():
            continue

        start_d = _parse_window_start_date(window_path)
        phase = (_quarter_index(start_d) - q0) % 4  # ←開始位置 mod4

        for m_key, m_info in meta.items():
            orig_name = m_info["orig"]
            target_csv = sub / f"予測値_水準ベース_{orig_name}.csv"
            if not target_csv.exists():
                continue

            pdf = pd.read_csv(target_csv, index_col=0, parse_dates=True)
            col_name = f"{orig_name}_Pred"
            if col_name not in pdf.columns:
                continue

            preds = pdf[col_name].dropna()
            if preds.empty:
                continue

            all_preds[m_key][phase].append(preds)

            # RMSE(そのwindow内)も保存
            actual = df_raw.loc[preds.index, orig_name].dropna()
            if not actual.empty:
                rmse = float(np.sqrt(np.mean((actual - preds.loc[actual.index])**2)))
                rmse_list[m_key][phase].append(rmse)

    # --- 描画（各変数ごと） ---
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for m_col, m_info in meta.items():
        orig_name = m_info["orig"]
        if not any(len(all_preds[m_col][p]) > 0 for p in range(4)):
            continue

        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        ax.set_facecolor('white')

        # ✅ 実績は「1回だけ」・前と同じ感じ（細め + 小さい丸）
        ax.plot(
            df_raw.index, df_raw[orig_name],
            color="#333333", lw=1.5,
            marker='o', markersize=3, alpha=0.7,
            label="実績値",
            zorder=2
        )

        rmse_summary = []
        for p in range(4):
            if not all_preds[m_col][p]:
                continue

            combined = pd.concat(all_preds[m_col][p]).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]

            avg_rmse = float(np.mean(rmse_list[m_col][p])) if rmse_list[m_col][p] else 0.0
            rmse_summary.append(f"P{p+1}:{avg_rmse:.3f}")

            # ✅ 予測も「前の感じ」(細め + 小さい丸) に戻す
            ax.plot(
                combined.index, combined.values,
                color=colors[p],
                linestyle='--',
                marker='o', markersize=3,   # ← 4.5 → 3
                lw=1.5, alpha=0.9,         # ← 2.6 → 2.0
                label=f"パターン{p+1} (RMSE: {avg_rmse:.3f})",
                zorder=5
            )

            combined.to_csv(csv_root / f"全期間連結予測_{model_subdir}_{orig_name}_P{p+1}.csv", encoding="utf-8-sig")

        vals = [np.mean(v) for v in rmse_list[m_col].values() if len(v) > 0]
        total_avg = float(np.mean(vals)) if vals else 0.0

        ax.set_title(
            f"全期間連結予測: {orig_name}\n【RMSE】{' / '.join(rmse_summary)} (平均: {total_avg:.3f})",
            fontsize=14, pad=20
        )
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white')
        ax.grid(True, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel("水準 (Level)")
        ax.set_xlabel("年月")

        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3,6,9,12]))
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25)
        ax.grid(True, which='minor', axis='x', linestyle=':',  alpha=0.10)

        plt.tight_layout()
        plt.savefig(save_root / f"全期間連結予測_{model_subdir}_{orig_name}.png", dpi=200, facecolor='white')
        plt.close()

    print(f"✅ PC1のみ（{model_subdir}）の全期間連結予測を保存しました")

def plot_exog_trends(csv_path, output_dir, target_var="GDP"):
    output_root = Path(output_dir)
    df = pd.read_csv(csv_path)

    sub_df = df[(df["Target_Variable"] == target_var) &
                (df["Model_Type"].str.contains("PC"))].copy()

    # --- 【ここがポイント：日付への変換】 ---
    # Window名の末尾（テスト期間の終了日）を日付型に変換する
    sub_df["EndDateStr"] = sub_df["Window"].str.split("_").str[-1]
    sub_df["Date"] = pd.to_datetime(sub_df["EndDateStr"], format='%Y%m%d')
    sub_df = sub_df.sort_values("Date")

    models = sub_df["Model_Type"].unique()
    pc_cols = [c for c in sub_df.columns if c.startswith("PC")]

    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]

    for ax, m_type in zip(axes, models):
        m_data = sub_df[sub_df["Model_Type"] == m_type]
        active_pcs = [c for c in pc_cols if not m_data[c].isna().all()]

        for col in active_pcs:
            # X軸に文字列ではなく「Date」オブジェクトを渡す
            ax.plot(m_data["Date"], m_data[col], marker='o', label=col, lw=2)

        ax.set_title(f"Target: {target_var} | Model: {m_type}", fontsize=12, fontweight='bold')
        ax.axhline(0.3, color='red', ls='--', alpha=0.6, label="LB: 0.3")
        ax.axhline(0, color='black', lw=1, alpha=0.3)

        # --- 【ここがポイント：X軸の目盛りを綺麗にする】 ---
        ax.xaxis.set_major_locator(mdates.YearLocator()) # 1年刻み
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m')) # 表示形式

        ax.set_ylabel("Coefficient")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='both', alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = output_root / f"係数推移_{target_var}_日付修正版.png"
    plt.savefig(save_path, dpi=200, facecolor='white')
    plt.close()
    print(f"✅ 日付を修正して保存しました: {save_path}")

# =========================================================
# 3. メイン処理
# =========================================================
def main():
    # データ読み込みと前処理（ここは共通）
    df_raw = pd.read_csv(CONFIG["data_path"], parse_dates=["Date"])
    df_raw = df_raw.set_index("Date").sort_index()
    sector_df, macro_df, common_idx, meta = prepare_aligned(df_raw)

    # --- Moving Window 設定 ---
    test_len = CONFIG["test_steps"]
    n_total = len(common_idx)
    all_coefficients_records=[]

    # テスト期間を1期ずつずらして全組み合わせを実行
    for i in range(n_total - test_len + 1):
        # テスト期間と訓練期間（テスト以外すべて）のインデックス作成
        te_idx_list = np.arange(i, i + test_len)
        tr_idx_list = np.setdiff1d(np.arange(n_total), te_idx_list)

        # フォルダ名用の日付取得
        d_start = common_idx[te_idx_list[0]]
        d_end = common_idx[te_idx_list[-1]]

        # ルートディレクトリ設定（ウィンドウごとに作成）
        root_dir = Path(CONFIG["output_dir"]) / f"Window_Test_{d_start:%Y%m%d}_{d_end:%Y%m%d}"
        root_dir.mkdir(parents=True, exist_ok=True)

        print(f"▶️ 実行中: {root_dir.name}")

        # =========================================================
        # PCA: 訓練データでFitし、全期間をTransform（PCは3固定）
        # =========================================================
        n_pcs = CONFIG["pc_max"]  # 常に3

        sc_pca = StandardScaler()
        X_tr_pca = sc_pca.fit_transform(sector_df.iloc[tr_idx_list])

        pca = PCA(n_components=n_pcs).fit(X_tr_pca)
        pc_cols_orig = [f"PC{k+1}" for k in range(n_pcs)]  # ["PC1","PC2","PC3"]

        # ★追加：ここで符号を確定（inplace）
        fix_pca_sign_inplace(
            pca=pca,
            feature_names=sector_df.columns,     # ← PCAに使った列順と一致してることが重要
            pc_cols=pc_cols_orig,
            sector_groups=SECTOR_GROUPS,
            top_n=5,
        )

        # 全期間のPCスコア（PC1..PC3）
        full_pcs = pca.transform(sc_pca.transform(sector_df))
        pc_all = pd.DataFrame(full_pcs, index=sector_df.index, columns=pc_cols_orig)

        # 差分（PC1..PC3）
        pc_diff_all = pc_all.diff().dropna()
        pc_cols_diff = [f"PC{k+1}_DIFF" for k in range(n_pcs)]  # ["PC1_DIFF","PC2_DIFF","PC3_DIFF"]
        pc_diff_all.columns = pc_cols_diff

        # --- 元の分析レポート(analyze_pca_details)実行 ---
        # 引数はWindowに合わせて調整
        analyze_pca_details(
            pca, sc_pca, sector_df, pc_cols_orig, root_dir,
            tr_idx_list=tr_idx_list, te_idx_list=te_idx_list
        )
        # マクロ同期
        m_df_win = macro_df.loc[pc_diff_all.index]
        combinations = [[]] + [list(c) for r in range(1, n_pcs+1) for c in itertools.combinations(pc_cols_diff, r)]
        summary_list = []

        # combinations 作った直後（for combo の前）に移動
        print("macro cols:", list(m_df_win.columns))
        print("combo sample:", combinations[:5], " ... total:", len(combinations))

        # 1. すべての制約をここに集約 (範囲指定 or 符号指定)
        STRICT_CONSTRAINTS = {
            ("PC1", "GDP"): (0.3, 0.7),
            ("PC1", "NIKKEI"): (0.3, 0.7),
            ("PC1", "CPI"): (0.1, 0.5),
            ("PC1", "UNEMP_RATE"): (-0.7, -0.3),
            # 以前の BASE_SIGN_CONSTRAINTS 相当もここに書いておけばエラーになりません
            # ("*", "UNEMP_RATE"): (-np.inf, -0.3), # 必要なら追加
        }

        for combo in combinations:
            m_name_display = ", ".join(combo) if combo else "マクロのみ"
            sub_dir = root_dir / ("_".join(combo) if combo else "BASE_VAR_ONLY")
            sub_dir.mkdir(parents=True, exist_ok=True)

            # --- 訓練とテストの切り出し ---
            win_idx = m_df_win.index
            tr_pre_dates = common_idx[tr_idx_list[tr_idx_list < te_idx_list[0]]]
            tr_post_dates = common_idx[tr_idx_list[tr_idx_list > te_idx_list[-1]]]
            te_dates = common_idx[te_idx_list]

            # 1. 過去側の訓練データ
            tr_pre = pd.concat([m_df_win.loc[win_idx.isin(tr_pre_dates)],
                                pc_diff_all.loc[win_idx.isin(tr_pre_dates), combo]], axis=1)

            # 2. 未来側の訓練データ
            tr_post = pd.concat([m_df_win.loc[win_idx.isin(tr_post_dates)],
                                 pc_diff_all.loc[win_idx.isin(tr_post_dates), combo]], axis=1)

            # 【重要】未来側データの先頭1行をNaNにする（ラグ計算で過去側と繋がるのを防ぐため）
            if not tr_post.empty:
                tr_post.iloc[0, :] = np.nan

            # 3. 訓練データの結合（中抜きされた箇所にNaNの壁ができる）
            tr_raw = pd.concat([tr_pre, tr_post])
            te_raw = pd.concat([m_df_win.loc[win_idx.isin(te_dates)],
                                pc_diff_all.loc[win_idx.isin(te_dates), combo]], axis=1)

            means, stds = tr_raw.mean(), tr_raw.std(ddof=0).replace(0, 1.0)
            tr_s, te_s = (tr_raw - means) / stds, (te_raw - means) / stds
            m_cols = list(m_df_win.columns)
            m_dim = len(m_cols)

            # モデル推定
            Y_tr, X_tr = make_design(tr_s[m_cols], tr_s[combo] if combo else None, CONFIG["p_lag"])

            if CONFIG.get("use_constraints", True):
                beta_list = []
                ridge_aug = np.sqrt(CONFIG["ridge"]) * np.eye(X_tr.shape[1])
                X_aug = np.vstack([X_tr, ridge_aug])

                # パラメータ名リスト
                param_names = ["CONST"] + [f"LAG_{m}" for m in m_cols] + list(combo)

                for j, target_m_col in enumerate(m_cols):
                    orig_target = meta[target_m_col]["orig"]
                    Y_aug = np.concatenate([Y_tr[:, j], np.zeros(X_tr.shape[1])])
                    lb = np.full(X_aug.shape[1], -np.inf)
                    ub = np.full(X_aug.shape[1], np.inf)

                    # --- 修正版：ワイルドカード対応の制約適用ロジック ---
                    if combo:
                        start_exog_idx = 1 + m_dim
                        for k, exog_pc_name in enumerate(combo):
                            orig_exog = exog_pc_name.split("_")[0]
                            target_idx = start_exog_idx + k

                            # 優先順位をつけて制約を探す
                            # 1. 個別指定 ("PC1", "GDP")
                            # 2. 外生変数全指定 ("*", "GDP")
                            # 3. ターゲット全指定 ("PC1", "*")
                            # 4. 全指定 ("*", "*")
                            bound = (STRICT_CONSTRAINTS.get((orig_exog, orig_target)) or
                                    STRICT_CONSTRAINTS.get(("*", orig_target)) or
                                    STRICT_CONSTRAINTS.get((orig_exog, "*")) or
                                    STRICT_CONSTRAINTS.get(("*", "*")))

                            if bound:
                                lb[target_idx] = bound[0]
                                ub[target_idx] = bound[1]
                            # --- エラーの元だった else 節 (old_c) は削除しました ---

                    res = lsq_linear(X_aug, Y_aug, bounds=(lb, ub), lsmr_tol='auto')
                    if not res.success:
                        # 保険：制約を無視して通常のridge（またはOLS）に落とす
                        res = lsq_linear(X_aug, Y_aug, bounds=(-np.inf, np.inf), lsmr_tol='auto')
                    beta_list.append(res.x)

                    record = {
                        "Window": root_dir.name,
                        "Model_Type": m_name_display,
                        "Target_Variable": orig_target,
                    }
                    # 各係数をカラムとして追加
                    for name, val in zip(param_names, res.x):
                        record[name] = val

                    # mainの冒頭で定義した all_coefficients_records に追加
                    all_coefficients_records.append(record)

                Beta = np.array(beta_list).T
            else:
                # 保険：制約を切った場合でも落ちないようにOLS
                Beta = np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0]

            # --- 既存のモデル推定直後から ---
            # 指標計算
            tr_resid = Y_tr - X_tr @ Beta
            n_obs = len(Y_tr)
            tr_rmse = np.sqrt(np.mean(tr_resid**2))

            Y_te, X_te = make_design(te_s[m_cols], te_s[combo] if combo else None, CONFIG["p_lag"])
            te_rmse = np.sqrt(np.mean((Y_te - X_te @ Beta)**2)) if len(Y_te) > 0 else np.nan

            # --- AIC計算ロジックの強化版 ---
            sigma_matrix = (tr_resid.T @ tr_resid) / n_obs
            sign, logdet = np.linalg.slogdet(sigma_matrix)

            if sign > 0:
                # 通常の計算
                aic_val = n_obs * logdet + 2 * Beta.size
            else:
                # 行列式が正常に計算できない場合のバックアップ
                # 正の固有値のみを抽出して対数和をとる
                eigs = np.linalg.eigvalsh(sigma_matrix)
                valid_eigs = eigs[eigs > 1e-15] # ほぼゼロ以上の固有値のみ
                if len(valid_eigs) > 0:
                    aic_val = n_obs * np.sum(np.log(valid_eigs)) + 2 * Beta.size
                else:
                    aic_val = np.nan
            # -------------------------------

            A1 = Beta[1:1+m_dim, :].T
            max_eig_abs = np.max(np.abs(np.linalg.eigvals(A1)))

            summary_list.append({
                "モデル構成": m_name_display,
                "訓練RMSE": round(tr_rmse, 4),
                "予測RMSE": round(te_rmse, 4) if not np.isnan(te_rmse) else np.nan,
                "RMSE比": round(te_rmse / tr_rmse, 2) if tr_rmse > 0 and not np.isnan(te_rmse) else np.nan,
                "AIC": round(aic_val, 2) if not np.isnan(aic_val) else np.nan,
                "最大固有値": round(max_eig_abs, 3)
            })

            # --- 予測結果の可視化 (逐次予測 & 全体表示版) ---
            # 1. 逐次予測 (Dynamic Forecast) の実行
            y_te_pred_list = []
            if len(X_te) > 0:
                # 最初の入力（テスト期間1点目のためのラグを含む）
                curr_X = X_te[0:1, :]

                for t in range(CONFIG["test_steps"]):
                    # 現在の入力で予測
                    pred_t = curr_X @ Beta  # shape: (1, m_dim)
                    y_te_pred_list.append(pred_t)

                    if t < CONFIG["test_steps"] - 1:
                        # 次の予測のための入力を構築
                        # [定数項(1), 今回の予測値(ラグ1), 外生変数(PC)の実際値]
                        next_lag_y = pred_t
                        # PC（外生変数）は実績がある範囲で使う（なければ0か最後の値）
                        if combo:
                            next_exog = te_s[combo].iloc[t+1:t+2].values if (t+1) < len(te_s) else te_s[combo].iloc[-1:].values
                        else:
                            next_exog = np.empty((1, 0))

                        curr_X = np.concatenate([[[1.0]], next_lag_y, next_exog], axis=1)

                y_te_pred_scaled = np.vstack(y_te_pred_list)
            else:
                y_te_pred_scaled = None

            # i=0 (最初期) を除外し、かつ予測値が存在する場合のみ実行
            if y_te_pred_scaled is not None and i != 0:

                # 1. 予測期間の日付インデックスを取得
                start_d_idx = np.where(common_idx == te_dates[0])[0][0]
                te_actual_dates = common_idx[start_d_idx : start_d_idx + CONFIG["test_steps"]]

                # 全期間表示（2008-
                # plot_start_idx = max(0, start_d_idx - 8)
                # plot_end_idx = min(len(common_idx), start_d_idx + CONFIG["test_steps"] + 4)
                # full_display_range = common_idx[plot_start_idx:plot_end_idx]    
                full_display_range = df_raw.index  # 実績はこれで全期間

                for i_m, m_col in enumerate(m_cols):
                    orig_name = meta[m_col]["orig"]
                    meth = meta[m_col]["method"]

                    # 3. 水準復元ロジック
                    # スケーリング解除
                    pred_change = (y_te_pred_scaled[:, i_m] * stds[m_col]) + means[m_col]
                    # i != 0 なので必ず直前の実績(ラグ)
                    hist_before = df_raw.loc[df_raw.index < te_actual_dates[0], orig_name].dropna()
                    if hist_before.empty:
                        continue
                    last_actual_level = hist_before.iloc[-1]

                    pred_levels = []
                    curr_level = last_actual_level
                    for val in pred_change:
                        if meth == "LOGDIFF":
                            curr_level = curr_level * np.exp(val)
                        elif meth == "DIFF":
                            curr_level = curr_level + val
                        elif meth == "LEVEL":
                            curr_level = val
                        else:
                            curr_level = curr_level + val
                        pred_levels.append(curr_level)

                    # --- 予測値の計算が終わった直後に配置 ---
                    p_df = pd.DataFrame(pred_levels, index=te_actual_dates, columns=[f"{orig_name}_Pred"])
                    # 変数名付きで保存することで、後の集計を確実にします
                    p_df.to_csv(sub_dir / f"予測値_水準ベース_{orig_name}.csv", encoding="utf-8-sig")

                    # --- 4. グラフ描画 ---
                    plt.figure(figsize=(11, 5.5))

                    # 実績値 (黒) と 予測値 (赤)
                    # 実績：薄め、小さめ
                    plt.plot(
                        full_display_range, df_raw.loc[full_display_range, orig_name],
                        label="実績値",
                        color="#333333", lw=1.5,
                        marker='o', markersize=3, alpha=0.7,
                        zorder=2
                    )
                    # 予測：太め、点大きめ、白縁で埋もれ防止、前面
                    plt.plot(
                        te_actual_dates, pred_levels,
                        label="予測値 (逐次)",
                        color="red", lw=1.5, linestyle="--",
                        marker="o", markersize=3,
                        markerfacecolor="red", markeredgecolor="red",
                        alpha=0.9, zorder=5
                    )
                    # テスト期間の背景
                    plt.axvspan(te_actual_dates[0], te_actual_dates[-1], color='gray', alpha=0.1, label='予測対象期間')

                    ax = plt.gca()
                    # --- x軸（長期向け）：年だけラベル、四半期は補助 ---
                    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))          # 1年刻み
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))        # "2008" みたいに年だけ

                    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3,6,9,12]))  # 四半期末は補助目盛り
                    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25)
                    ax.grid(True, which='minor', axis='x', linestyle=':',  alpha=0.10)
                    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.25)

                    plt.xticks(rotation=0)  # 年だけなら回転なしでOK
                    plt.title(f"【予測】{orig_name} : {m_name_display}", fontsize=14)
                    plt.xlabel("年月", fontsize=12)
                    plt.ylabel("水準 (Level)", fontsize=12)
                    plt.legend(loc='best', frameon=True, shadow=True)
                    plt.tight_layout()

                    # 保存
                    plt.savefig(sub_dir / f"PRED_{orig_name}.png", dpi=150)
                    plt.close()

            # --- 元のIRFプロット(変更なし) ---
            if CONFIG["do_irf"] and combo:
                B = Beta[1+m_dim:, :].T
                for j, pc_label in enumerate(combo):
                    pc_folder = sub_dir / f"Shock_{pc_label}"; pc_folder.mkdir(parents=True, exist_ok=True)
                    impact = np.zeros((13, m_dim)); impact[1] = B[:, j]
                    for h in range(2, 13): impact[h] = A1 @ impact[h-1]
                    for i_m, m_col in enumerate(m_cols):
                        orig_m, meth_m = meta[m_col]["orig"], meta[m_col]["method"]
                        base = BASE_LEVELS.get(orig_m, 100); imp_raw = impact[:, i_m] * stds[m_col]
                        vals = [base]
                        for s in range(1, 13):
                            if meth_m == "LOGDIFF":
                                vals.append(vals[-1] * np.exp(imp_raw[s]))
                            elif meth_m == "DIFF":
                                vals.append(vals[-1] + imp_raw[s])
                            elif meth_m == "LEVEL":
                                vals.append(base + imp_raw[s])  # LEVELは“基準値＋水準ショック”
                            else:
                                vals.append(vals[-1] + imp_raw[s])
                        plt.figure(figsize=(6, 3.5))
                        ax = plt.gca()
                        ax.yaxis.get_major_formatter().set_useOffset(False)
                        ax.yaxis.get_major_formatter().set_scientific(False)
                        plt.plot(range(13), vals, color='k', lw=1)
                        plt.scatter(range(13), vals, marker='o', s=15, color='k', zorder=2)
                        plt.scatter(1, vals[1], facecolors='none', edgecolors='k', marker='o', s=100, lw=1, zorder=3)
                        plt.axvline(x=1, color='k', ls='--', lw=0.7, alpha=0.6)
                        plt.axhline(base, color='gray', ls=':', lw=0.8)
                        plt.title(f"{pc_label} → {orig_m}")
                        plt.grid(alpha=0.15); plt.tight_layout()
                        plt.savefig(pc_folder / f"{orig_m}_応答.png"); plt.close()

        # ウィンドウごとのサマリー保存
        pd.DataFrame(summary_list).sort_values("AIC").to_csv(root_dir / "モデル比較サマリー.csv", index=False, encoding="utf-8-sig")

    # --- 実行部分 ---
    agg_root = Path(CONFIG["output_dir"]) / "00_aggregate"
    for d in ["model_eval", "pca", "heatmaps", "forecasts", "coefficients"]:
        (agg_root / d).mkdir(parents=True, exist_ok=True)

    aggregate_results(Path(CONFIG["output_dir"]), agg_root=agg_root)

    csv_file = agg_root / "model_eval" / "全ウィンドウ集計_モデル評価_詳細版.csv"
    if csv_file.exists():
        visualize_model_performance(csv_file)

    plot_connected_forecasts(CONFIG["output_dir"], agg_root / "forecasts", df_raw, meta, "PC1_DIFF")
    plot_connected_forecasts(CONFIG["output_dir"], agg_root / "forecasts", df_raw, meta, "BASE_VAR_ONLY")

    if all_coefficients_records:
        df_coef = pd.DataFrame(all_coefficients_records)
        coef_path = agg_root / "coefficients" / "all_model_coefficients.csv"
        df_coef.to_csv(coef_path, index=False, encoding="utf-8-sig")

        for t in ["GDP", "NIKKEI", "USD_JPY"]:
            plot_exog_trends(coef_path, agg_root / "coefficients", target_var=t)

    print(f"✅ 全工程完了。{agg_root} を確認してください。")

if __name__ == "__main__":
    main()
