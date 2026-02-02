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
import os
import platform
from matplotlib import font_manager as fm

warnings.simplefilter('ignore')

# =========================================================
# 1. 設定・定数
# =========================================================
CONFIG = {
    "data_path": "./data/all_q_merged_new_tmp.csv",
    "output_dir": "./output_mw_new_tmp_5",
    "test_steps": 4,
    "pc_max": 3,
    "p_lag": 1,
    "ridge": 1.0,
    "do_irf": True,
    "verbose": False,
    "transform_overrides": {
        # "CPI": "LEVEL",   # 前年比%をそのまま使うなら
        # "CPI": "DIFF",    # 前年比%の前年差なら
        # "CPI": "LOGDIFF", # CPI指数(>0)なら
        # "UNEMP_RATE": "LOGDIFF" 
    }
}

BASE_LEVELS = {
    "GDP": 500000, "NIKKEI": 20000, "USD_JPY": 150, "UNEMP_RATE": 3.0,
    "JGB_1Y": 0.0, "JGB_3Y": 0.0,
    # "JGB_10Y": 0.0,
    "CPI": 0.0, 
    # "TOPIX": 1500,
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

# =========================================================
# 2. ロジック関数
# =========================================================
def setup_japanese_font(prefer_path=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    from pathlib import Path
    import os, platform

    candidates = []
    if prefer_path:
        candidates.append(prefer_path)

    envp = os.environ.get("JP_FONT_PATH")
    if envp:
        candidates.append(envp)

    sys = platform.system()
    if sys == "Windows":
        candidates += [
            r"C:\Windows\Fonts\YuGothR.ttc",
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
        ]
    elif sys == "Darwin":
        candidates += [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
    else:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",  # 環境によって
        ]

    for p in candidates:
        p = str(p)
        if p and Path(p).exists():
            fm.fontManager.addfont(p)
            fp = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [fp.get_name()]
            mpl.rcParams["axes.unicode_minus"] = False
            print(f"✅ 日本語フォント: {fp.get_name()} ({p})")
            return

    # ここに来たら「そもそも日本語フォントが無い」
    mpl.rcParams["axes.unicode_minus"] = False
    print("⚠️ 日本語フォントが見つからず、文字化けの可能性があります。JP_FONT_PATH で .ttf/.ttc を指定してください。")

def apply_style_and_jpfont(style_name="default", prefer_path=None):
    plt.style.use(style_name)          # ← 先にstyle
    setup_japanese_font(prefer_path)   # ← 後でフォント（これが重要）

def block_starts(s):
    # 予測が始まる点（NaN→値あり）
    return s.index[s.notna() & s.shift(1).isna()]

def make_full_pred(pred_by_h):  # {1:Series, 2:Series, 3:Series, 4:Series}
    full = pred_by_h[1].copy()
    for h in [2, 3, 4]:
        full = full.combine_first(pred_by_h[h])
    return full

def log(*args, **kwargs):
    if CONFIG.get("verbose", True):
        print(*args, **kwargs)

def smart_transform(series, name):
    overrides = CONFIG.get("transform_overrides", {})
    mode = overrides.get(name)

    if mode == "LEVEL":
        return series, "LEVEL"
    if mode == "DIFF":
        return series.diff(), "DIFF"
    if mode == "LOGDIFF":
        s = series.copy()
        s = s.where(s > 0, np.nan)  # 0以下をNaN
        return np.log(s).diff(), "LOGDIFF"
    if mode == "PCTCHANGE":
        # pct_change は 0割や inf が出やすいので一応ケア
        ts = series.pct_change().replace([np.inf, -np.inf], np.nan)
        return ts, "PCTCHANGE"

    # ---- デフォルト（現状の自動判定）----
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
    keep = [c for c in m_work.columns if c.startswith(("GDP_", "NIKKEI_", "USD_JPY_", "UNEMP_RATE_"))]
    m_df = m_work.dropna(subset=keep)
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

    idx_valid = endog.index[valid]
    return Y[valid], X[valid], idx_valid

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

    # ここを ggplot から戻す（灰背景の元）
    apply_style_and_jpfont("default")

    if "AIC_平均" in df.columns and "予測RMSE_平均" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
        ax.set_facecolor("white")

        # seaborn scatter
        sc = sns.scatterplot(
            data=df,
            x="AIC_平均",
            y="予測RMSE_平均",
            size="予測RMSE_標準偏差" if "予測RMSE_標準偏差" in df.columns else None,
            hue=df.index,              # モデル構成
            sizes=(100, 1000),
            alpha=0.75,
            ax=ax
        )

        # 点ラベル（必要なら）
        for i, txt in enumerate(df.index):
            ax.annotate(
                txt,
                (df["AIC_平均"].iloc[i], df["予測RMSE_平均"].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

        ax.set_title("モデルの複雑さ(AIC) vs 予測精度(RMSE)")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("AIC_平均")
        ax.set_ylabel("予測RMSE_平均")

        # -----------------------------
        # 凡例を「色(hue)」と「サイズ(size)」で分離して右外へ
        # -----------------------------
        handles, labels = ax.get_legend_handles_labels()

        # seaborn の凡例は [hueタイトル, hue項目..., sizeタイトル, size項目...] の順になりがち
        # ここを安定に分割するため、タイトル文字で区切る
        hue_title = "モデル構成"
        size_title = "予測RMSE_標準偏差"

        def split_legend(handles, labels, title_a, title_b):
            # title_a が出てくる位置、title_b が出てくる位置を探す
            ia = labels.index(title_a) if title_a in labels else None
            ib = labels.index(title_b) if title_b in labels else None
            if ia is None:
                return None
            if ib is None:
                # hue しか無い
                return (handles[ia+1:], labels[ia+1:], None, None)

            hue_h = handles[ia+1:ib]
            hue_l = labels[ia+1:ib]
            size_h = handles[ib+1:]
            size_l = labels[ib+1:]
            return (hue_h, hue_l, size_h, size_l)

        sp = split_legend(handles, labels, hue_title, size_title)

        # 既存の一体化凡例を消す
        leg0 = ax.get_legend()
        if leg0 is not None:
            leg0.remove()

        # hue 凡例（上）
        if sp is not None:
            hue_h, hue_l, size_h, size_l = sp

            leg1 = ax.legend(
                hue_h, hue_l,
                title=hue_title,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.00),
                borderaxespad=0.,
                frameon=True
            )
            ax.add_artist(leg1)

            # size 凡例（下）
            if size_h is not None and len(size_h) > 0:
                ax.legend(
                    size_h, size_l,
                    title=size_title,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 0.55),
                    borderaxespad=0.,
                    frameon=True
                )
        else:
            # 万一分割できない時は、右外へだけ出す
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)

        # 右に凡例スペース確保
        fig.tight_layout(rect=[0, 0, 0.78, 1])

        save_path = output_dir / "全期間_AIC_vs_RMSE_分布.png"
        fig.savefig(save_path, dpi=300, facecolor="white")
        plt.close(fig)

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
    apply_style_and_jpfont("default")
    window_root = Path(window_root)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # CSV出力先（モデル別）
    csv_root = save_root / "csv" / model_subdir
    csv_root.mkdir(parents=True, exist_ok=True)

    # 追加：画像出力先（モデル別）
    fig_root = save_root / "fig" / model_subdir
    fig_root.mkdir(parents=True, exist_ok=True)

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
        phase = (_quarter_index(start_d) - q0) % 4  # 開始位置 mod4

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

    # --- 描画＆保存（各変数ごと） ---
    for m_col, m_info in meta.items():
        orig_name = m_info["orig"]

        # 4パターンのどれも無ければスキップ
        if not any(len(all_preds[m_col][p]) > 0 for p in range(4)):
            continue

        # 実績
        actual = df_raw[orig_name].copy()
        full_index = actual.index

        # phase(P1..P4)の連結Seriesを作る → full_index に reindex して NaN で切れ目を作る
        pred_by_h = {}
        rmse_summary = []

        for p in range(4):
            if all_preds[m_col][p]:
                combined = pd.concat(all_preds[m_col][p]).sort_index()
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.reindex(full_index)  # ★ギャップはNaNになる（線が勝手に切れる）
            else:
                combined = pd.Series(np.nan, index=full_index)

            pred_by_h[p+1] = combined

            avg_rmse = float(np.mean(rmse_list[m_col][p])) if rmse_list[m_col][p] else 0.0
            rmse_summary.append(f"P{p+1}:{avg_rmse:.3f}")

            # 既存と同じく、各色CSVも保存（dropnaして保存）
            out_p_csv = csv_root / f"全期間連結予測_{model_subdir}_{orig_name}_P{p+1}.csv"
            combined.dropna().to_frame(f"{orig_name}_Pred").to_csv(out_p_csv, encoding="utf-8-sig")

        # 全期間予測（一本）を作る（P1優先→P2→P3→P4で穴埋め）
        full_pred = make_full_pred(pred_by_h)

        # 予測が存在する期間だけ灰色シェード（任意）
        shade = None
        if full_pred.notna().any():
            shade = (full_pred.first_valid_index(), full_pred.last_valid_index())

        # --- (1) 4色まとめ + 全期間一本 + 開始点丸 ---
        title = f"全期間連結予測: {orig_name}\n【RMSE】{' / '.join(rmse_summary)}"
        out_png = fig_root / f"全期間連結予測_{model_subdir}_{orig_name}_ALL.png"
        plot_forecasts_all(actual, pred_by_h, out_png, title, shade=shade)

        # --- (2) 色ごとに1枚ずつ（開始点丸入り） ---
        each_dir = fig_root / f"{orig_name}_each"
        base_title = f"全期間連結予測: {orig_name} | {model_subdir}"
        plot_forecast_each(actual, pred_by_h, each_dir, base_title, shade=shade)

        # --- (3) CSVもPred_Full付きで保存 ---
        out_all_csv = csv_root / f"全期間連結予測_{model_subdir}_{orig_name}_ALL.csv"
        export_forecast_csv(actual, pred_by_h, out_all_csv)

    print(f"✅ 全期間連結予測（ALL + 各P + 開始点丸 + Pred_Full CSV）を保存しました: {fig_root}")

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

def splice_points_every_k(s: pd.Series, k: int = 4):
    """
    s: 予測Series（NaNでギャップが入る想定）
    連続区間ごとに「先頭 + k点ごと」のindexを返す
    例: k=4 なら 0,4,8,... (=赤丸→3点→赤丸…)
    """
    idxs = []

    # 非NaNの位置だけ取り出し
    valid = s.dropna()
    if valid.empty:
        return valid.index[:0]

    # 元のindex上で「連続区間」を検出（NaNで切れている前提）
    is_valid = s.notna()
    # 連続区間ID（NaN→値ありで+1）
    run_id = (is_valid & ~is_valid.shift(1, fill_value=False)).cumsum()
    run_id = run_id.where(is_valid)

    for rid, part in s[is_valid].groupby(run_id[is_valid]):
        part = part.sort_index()
        # 0, k, 2k, ... 番目を接着点として採用
        take = part.iloc[::k]
        idxs.extend(list(take.index))

    return pd.Index(idxs)

def plot_forecasts_all(actual, pred_by_h, out_png, title, shade=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    # 実績：黒
    ax.plot(actual.index, actual.values, linewidth=2.2, color="black", label="実績値", zorder=2)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # P1..P4

    for h in [1, 2, 3, 4]:
        s = pred_by_h[h]
        ax.plot(
            s.index, s.values,
            linestyle="--", marker="o", markersize=3,
            color=colors[h-1],
            label=f"予測 P{h}", zorder=4
        )

        # 接着点（4点ごと）※凡例には出さない
        sp = splice_points_every_k(s, k=4)
        ax.scatter(
            sp, s.loc[sp],
            s=170, facecolors="none",
            edgecolors=colors[h-1], linewidths=2.6,
            zorder=6
        )

    # 予測範囲：グレー（凡例は1回だけ）
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color="lightgray", alpha=0.25, label="予測対象期間", zorder=1)

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10,1]))
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25)
    ax.grid(True, which='minor', axis='x', linestyle=':',  alpha=0.10)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("年月")
    ax.set_ylabel("水準 (Level)")
    ax.grid(True, alpha=0.3)

    # ★凡例をさらにコンパクトに（列数を増やして横に伸ばす）
    ax.legend(loc="upper left", ncol=3, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_forecast_each(actual, pred_by_h, out_dir, base_title, shade=None):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for h in [1, 2, 3, 4]:
        s = pred_by_h[h]

        fig, ax = plt.subplots(figsize=(14, 5))

        # 実績は黒
        ax.plot(actual.index, actual.values, linewidth=2.2, color="black", label="実績値", zorder=2)

        # 予測（Pごと）
        ax.plot(
            s.index, s.values,
            color=colors[h-1], linestyle="--",
            marker="o", markersize=3,
            label=f"予測 P{h}", zorder=4
        )

        # 開始点（色は線と同じ）
        sp = splice_points_every_k(s, k=4)   # ← 4点ごと（0,4,8,...）/区間ごと

        ax.scatter(
            sp, s.loc[sp],
            s=160, facecolors="none",
            edgecolors=colors[h-1], linewidths=2.4,
            label="接着点（4点ごと）", zorder=6
        )

        # 予測対象期間の背景（グレー）
        if shade is not None:
            ax.axvspan(shade[0], shade[1], color="lightgray", alpha=0.25, label="予測対象期間", zorder=1)

        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10,1]))
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.25)
        ax.grid(True, which='minor', axis='x', linestyle=':',  alpha=0.10)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        ax.set_title(f"{base_title} (P{h})")
        ax.set_xlabel("年月")
        ax.set_ylabel("水準 (Level)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"forecast_P{h}.png", dpi=200)
        plt.close(fig)

def export_forecast_csv(actual, pred_by_h, out_csv):
    full_pred = make_full_pred(pred_by_h)
    df_out = actual.to_frame("Actual")
    df_out["Pred_Full"] = full_pred
    for h in [1,2,3,4]:
        df_out[f"Pred_P{h}"] = pred_by_h[h]
    df_out.to_csv(out_csv, encoding="utf-8-sig")

# =========================================================
# 3. メイン処理
# =========================================================
def main():
    # setup_japanese_font(r"C:\Windows\Fonts\YuGothR.ttc")
    apply_style_and_jpfont("default")
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
        pc_diff_all = pc_all.diff()
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
            ("PC1", "GDP"): (0.2, np.inf),
            ("PC1", "NIKKEI"): (0.2, np.inf),
            ("PC1", "UNEMP_RATE"): (-np.inf, -0.1),
            ("PC2", "GDP"): (0.05, np.inf),
            ("PC2", "NIKKEI"): (0.05, np.inf),
            ("PC2", "UNEMP_RATE"): (-np.inf, -0.05),
            # 以前の BASE_SIGN_CONSTRAINTS 相当もここに書いておけばエラーになりません
            # ("*", "UNEMP_RATE"): (-np.inf, -0.3), # 必要なら追加
        }

        LAG_CONSTRAINTS = {
            # (target, pred): (lb, ub)
            # 例）失業率方程式で、GDPラグは負
            # ("UNEMP_RATE", "GDP"): (100, np.inf),

            # ワイルドカードもOK
            # ("*", "JGB_10Y"): (0.0, 0.0),   # 例：JGB_10Yのラグ効果を全部ゼロ
        }

        for combo in combinations:
            m_name_display = ", ".join(combo) if combo else "マクロのみ"
            sub_dir = root_dir / ("_".join(combo) if combo else "BASE_VAR_ONLY")
            sub_dir.mkdir(parents=True, exist_ok=True)

            # --- 訓練とテストの切り出し ---
            win_idx = m_df_win.index
            te_dates = win_idx[te_idx_list]   # ← これで必ず index 内
            tr_pre_dates = common_idx[tr_idx_list[tr_idx_list < te_idx_list[0]]]
            tr_post_dates = common_idx[tr_idx_list[tr_idx_list > te_idx_list[-1]]]

            # 1. 過去側の訓練データ
            tr_pre = pd.concat([m_df_win.loc[win_idx.isin(tr_pre_dates)],
                                pc_diff_all.loc[win_idx.isin(tr_pre_dates), combo]], axis=1)

            # 2. 未来側の訓練データ
            tr_post = pd.concat([m_df_win.loc[win_idx.isin(tr_post_dates)],
                                 pc_diff_all.loc[win_idx.isin(tr_post_dates), combo]], axis=1)

            te_raw = pd.concat([
                m_df_win.loc[win_idx.isin(te_dates)],
                pc_diff_all.loc[win_idx.isin(te_dates), combo]
            ], axis=1)
            # 【重要】未来側データの先頭1行をNaNにする（ラグ計算で過去側と繋がるのを防ぐため）
            tr_raw_stats = pd.concat([tr_pre, tr_post])  # 壁なし（統計は自然に）
            tr_post_design = tr_post.copy()
            if not tr_post_design.empty:
                tr_post_design.iloc[0, :] = np.nan       # 壁あり（設計行列のみ）
            tr_raw_design = pd.concat([tr_pre, tr_post_design])

            means = tr_raw_stats.mean()
            stds  = tr_raw_stats.std(ddof=0).replace(0, 1.0)

            tr_s = (tr_raw_design - means) / stds
            te_s = (te_raw        - means) / stds
            m_cols = list(m_df_win.columns)
            m_dim = len(m_cols)

            gdp_col = next(k for k in m_cols if meta[k]["orig"] == "GDP")
            gdp_j = m_cols.index(gdp_col)

            # モデル推定
            Y_tr, X_tr, idx_tr = make_design(tr_s[m_cols], tr_s[combo] if combo else None, CONFIG["p_lag"])

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

                    # =============================
                    # A(ラグ係数)の制約（A1の要素制約）
                    # =============================
                    # lag係数ブロックは param_names の ["CONST"] の次なので index=1 から始まる
                    for pred_i, pred_m_col in enumerate(m_cols):
                        pred_orig = meta[pred_m_col]["orig"]     # 例: "GDP"
                        target_idx = 1 + pred_i                  # CONSTの次がLAG群

                        bound = (LAG_CONSTRAINTS.get((orig_target, pred_orig)) or
                                LAG_CONSTRAINTS.get(("*", pred_orig)) or
                                LAG_CONSTRAINTS.get((orig_target, "*")) or
                                LAG_CONSTRAINTS.get(("*", "*")))

                        if bound:
                            lb[target_idx] = bound[0]
                            ub[target_idx] = bound[1]
                            print("APPLY LAG CONSTRAINT",
                            "target=", orig_target,
                            "pred=", pred_orig,
                            "idx=", target_idx,
                            "lb,ub=", bound)

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
            # ===== trainの fitted（標準化空間→元スケールへ）=====
            fitted_scaled = X_tr @ Beta
            means_vec = means[m_cols].values
            stds_vec  = stds[m_cols].values

            fitted_change = fitted_scaled * stds_vec + means_vec
            fitted_change_df = pd.DataFrame(fitted_change, index=idx_tr, columns=m_cols)

            # ===== レベルに復元（当てはめ値）=====
            # 1) idx_tr 上で fitted_level_df を作る
            fitted_level_df = pd.DataFrame(index=idx_tr)

            for m_col in m_cols:
                orig_name = meta[m_col]["orig"]
                meth      = meta[m_col]["method"]

                if meth == "LEVEL":
                    fitted_level = fitted_change_df[m_col]
                else:
                    prev_actual = df_raw[orig_name].shift(1).reindex(idx_tr)
                    chg = fitted_change_df[m_col]
                    if meth == "LOGDIFF":
                        fitted_level = prev_actual * np.exp(chg)
                    elif meth == "DIFF":
                        fitted_level = prev_actual + chg
                    elif meth == "PCTCHANGE":
                        fitted_level = prev_actual * (1.0 + chg)
                    else:
                        fitted_level = prev_actual + chg

                fitted_level_df[f"{orig_name}_Fitted"] = fitted_level

            # 2) 全期間 index に埋め戻し
            fitted_level_df_full = pd.DataFrame(index=m_df_win.index)
            for m_col in m_cols:
                orig_name = meta[m_col]["orig"]
                col = f"{orig_name}_Fitted"
                fitted_level_df_full[col] = np.nan
                fitted_level_df_full.loc[idx_tr, col] = fitted_level_df[col].values

            # 3) テスト期間は NaN で切断
            fitted_level_df_full.loc[fitted_level_df_full.index.intersection(te_dates), :] = np.nan

            # 指標計算
            tr_resid = Y_tr - X_tr @ Beta
            n_obs = len(Y_tr)
            tr_rmse = float(np.sqrt(np.nanmean(tr_resid**2)))

            # ===== 追加：変数別RMSE（train）=====
            tr_rmse_by = np.sqrt(np.nanmean(tr_resid**2, axis=0))  # (m_dim,)

            # 例：GDPのtrain RMSEだけ抜
            tr_rmse_gdp = float(tr_rmse_by[gdp_j])

            # --- テストの直前1期をコンテキストとして追加（p_lag=1想定） ---
            p = CONFIG["p_lag"]

            # テスト開始日の直前日（common_idx上）
            te_start_date = te_dates[0]
            pos = np.where(common_idx == te_start_date)[0][0]
            if pos - p < 0:
                # 最初すぎる場合は予測不可
                Y_te = X_te = idx_te = None
            else:
                ctx_dates = common_idx[pos-p:pos]  # 直前p期
                # 標準化済み series を作る（m_cols と combo を含む形）
                ctx_raw = pd.concat([
                    m_df_win.loc[ctx_dates, m_cols],
                    pc_diff_all.loc[ctx_dates, combo] if combo else pd.DataFrame(index=ctx_dates)
                ], axis=1)

                # ctx も train の means/stds で標準化（ここ重要）
                ctx_s = (ctx_raw - means) / stds

                # コンテキスト + テスト を縦結合して design 作る
                need_cols = m_cols + list(combo)
                ctx_s = ctx_s.reindex(columns=need_cols)
                te_s  = te_s.reindex(columns=need_cols)
                te_raw2 = pd.concat([ctx_s, te_s], axis=0)

                Y_te, X_te, idx_te = make_design(te_raw2[m_cols], te_raw2[combo] if combo else None, p)

                # ここで idx_te は「コンテキストを含んだ後の valid 行」なので、
                # 予測対象をテスト部分に絞る
                is_test = idx_te.isin(te_dates)
                Y_te = Y_te[is_test]
                X_te = X_te[is_test]
                idx_te = idx_te[is_test]
            te_rmse_gdp = np.nan
            if Y_te is None or X_te is None or idx_te is None or len(Y_te) == 0:
                te_rmse = np.nan
            else:
                te_resid = Y_te - X_te @ Beta
                te_rmse = float(np.sqrt(np.nanmean(te_resid**2)))
                # ===== 追加：変数別RMSE（test）=====
                te_rmse_by = np.sqrt(np.nanmean(te_resid**2, axis=0))  # (m_dim,)

                # 例：GDPのtest RMSEだけ抜
                te_rmse_gdp = float(te_rmse_by[gdp_j])

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
                "訓練RMSE_GDP": round(tr_rmse_gdp, 4),
                "予測RMSE": round(te_rmse, 4) if not np.isnan(te_rmse) else np.nan,
                "予測RMSE_GDP": round(te_rmse_gdp, 4),
                "RMSE比": round(te_rmse / tr_rmse, 2) if tr_rmse > 0 and not np.isnan(te_rmse) else np.nan,
                "AIC": round(aic_val, 2) if not np.isnan(aic_val) else np.nan,
                "最大固有値": round(max_eig_abs, 3)
            })

            # --- 予測結果の可視化 (逐次予測 & 全体表示版) ---
            # 1. 逐次予測 (Dynamic Forecast) の実行
            y_te_pred_list = []
            if X_te is not None and len(X_te) > 0:
                curr_X = X_te[0:1, :]

                for t in range(len(idx_te)):  # ← test_steps ではなく idx_te に合わせる
                    pred_t = curr_X @ Beta
                    y_te_pred_list.append(pred_t)

                    if t < len(idx_te) - 1:
                        next_lag_y = pred_t

                        if combo:
                            # 次の時点の exog を idx_te[t+1] で取る（安全）
                            d_next = idx_te[t+1]
                            next_exog = te_s.loc[[d_next], combo].values
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
                te_actual_dates = idx_te

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
                        elif meth == "PCTCHANGE":
                            curr_level = curr_level * (1.0 + val)
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
                    # 追加：当てはめ（trainの1期先当てはめ）
                    plt.plot(
                        fitted_level_df_full.index,
                        fitted_level_df_full[f"{orig_name}_Fitted"],
                        label="当てはめ値 (1期先)",
                        color="#1f77b4", lw=1.5, linestyle="--",
                        marker="o", markersize=3,
                        alpha=0.8, zorder=3
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
                            elif meth_m == "PCTCHANGE":
                                vals.append(vals[-1] * (1.0 + imp_raw[s]))
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
