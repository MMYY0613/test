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

warnings.simplefilter('ignore')

# =========================================================
# 1. 設定・定数
# =========================================================
CONFIG = {
    "data_path": "./data/all_q_merged.csv",
    "output_dir": "./output_pca_final",
    "train_range": ("2015-01-01", "2016-06-30"),
    "test_steps": 1,
    "pc_max": 3,
    "p_lag": 1,
    "ridge": 1.0,
    "do_irf": True,
}

BASE_LEVELS = {
    "GDP": 500000, "NIKKEI": 20000, "USD_JPY": 110, "UNEMP_RATE": 3.0,
    "JGB_1Y": 0.1, "JGB_2Y": 0.2, "JGB_3Y": 0.3, "CPI": 100
}

TARGET_MACRO = list(BASE_LEVELS.keys())

SECTOR_COLS = [
    "RET_FOODS", "RET_ENERGY_RESOURCES", "RET_CONSTRUCTION_MATERIALS", "RET_RAW_MAT_CHEM",
    "RET_PHARMACEUTICAL", "RET_AUTOMOBILES_TRANSP_EQUIP", "RET_STEEL_NONFERROUS",
    "RET_MACHINERY", "RET_ELEC_APPLIANCES_PRECISION", "RET_IT_SERV_OTHERS",
    "RET_ELECTRIC_POWER_GAS", "RET_TRANSPORT_LOGISTICS", "RET_COMMERCIAL_WHOLESALE",
    "RET_RETAIL_TRADE", "RET_BANKS", "RET_FIN_EX_BANKS", "RET_REAL_ESTATE", "RET_TEST"
]

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = ["Hiragino Sans"]

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
    s_raw = df_raw[[c for c in SECTOR_COLS if c in df_raw.columns]].interpolate(limit_direction='both')
    s_diff = s_raw.dropna() if all(c.startswith("RET_") for c in s_raw.columns) else s_raw.diff().dropna()
    common = s_diff.index.intersection(m_df.index)
    return s_diff.loc[common], m_df.loc[common], common, meta

def pca_no_leak(X_all, t_start, t_end, t_size, k_use=3):
    X_tr = X_all.iloc[t_start:t_end]
    X_te = X_all.iloc[t_end:t_end+t_size]
    sc = StandardScaler(); pca = PCA(n_components=k_use)
    Ztr = pca.fit_transform(sc.fit_transform(X_tr))
    Zte = pca.transform(sc.transform(X_te))
    cols = [f"PC{i+1}" for i in range(k_use)]
    pc_df = pd.concat([pd.DataFrame(Ztr, index=X_tr.index, columns=cols),
                       pd.DataFrame(Zte, index=X_te.index, columns=cols)])
    return pc_df.diff().dropna(), pca.explained_variance_ratio_

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

def analyze_pca_details(pca, sector_df, pc_cols, root_dir, t_start, t_end, t_size):
    """PCA分析：標準化データからの生スコア算出・プロット・負荷量CSV出力（軸ラベル追加版）"""
    pca_dir = root_dir / "pca_analysis"
    pca_dir.mkdir(exist_ok=True)
    
    # 訓練+テスト期間のデータを抽出して標準化
    X_target = sector_df.iloc[t_start : t_end + t_size]
    sc = StandardScaler()
    sc.fit(sector_df.iloc[t_start : t_end])
    X_scaled = sc.transform(X_target)
    
    # 主成分スコアの算出
    scores = pca.transform(X_scaled)
    score_df = pd.DataFrame(scores, index=X_target.index, columns=pc_cols)
    
    # スコアのCSV出力
    score_df.to_csv(pca_dir / "主成分スコア_生データ.csv", encoding="utf-8-sig")

    expl = pca.explained_variance_ratio_
    components = pca.components_.copy()
    
    # 符号の反転処理（解釈を容易にするため）
    for i in range(components.shape[0]):
        if np.mean(components[i]) < 0:
            components[i] *= -1
            score_df.iloc[:, i] *= -1 

    # --- 負荷量（Loadings）のCSV出力 ---
    loadings = pd.DataFrame(components.T, index=sector_df.columns, columns=pc_cols)
    loadings.to_csv(pca_dir / "セクターの負荷量_一覧.csv", encoding="utf-8-sig")

    pca_dir = root_dir / "pca_analysis"
    pca_dir.mkdir(exist_ok=True)

    # --- 寄与率の保存を追加 ---
    expl = pca.explained_variance_ratio_
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

def aggregate_results(output_root):
    output_root = Path(output_root)
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
                    abs_row = ld_df[pc].abs().to_frame().T
                    abs_row.index = [f"{window_name}_{pc}"]
                    abs_row.insert(0, "Window", window_name)
                    abs_row.insert(1, "PC_Type", pc)
                    abs_row.insert(2, "Explained_Variance", vr_df.at["ExplainedVariance", pc])
                    pca_abs_details.append(abs_row)

    if not all_summaries:
        return

    # --- 2. モデル評価の詳細統計（忘れてないポイント：0除外 & 日本語カラム） ---
    # ignore_index=True を追加して、インデックスの重複を解消する
    merged_summary = pd.concat(all_summaries, ignore_index=True)

    plot_final_boxplots(merged_summary, output_root)
    
    # 箱ひげ図を生データから作成
    # plot_model_boxplots(merged_summary, output_root)
    
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
    model_perf_detail.to_csv(output_root / "全ウィンドウ集計_モデル評価_詳細版.csv", encoding="utf-8-sig")

    # --- 3. PCA構造の集計とヒートマップ（ここも忘れてません！） ---
    if pca_abs_details:
        full_pca_df = pd.concat(pca_abs_details)
        full_pca_df.to_csv(output_root / "全ウィンドウ集計_PCA構造_絶対値詳細.csv", encoding="utf-8-sig")
        # 絶対値ヒートマップを保存
        plot_abs_heatmaps(full_pca_df, output_root)

    print(f"✅ すべての集計・画像保存（ヒートマップ、箱ひげ図、統計詳細）が完了しました。")

def plot_custom_error_boxes(merged_summary, output_root):
    """
    【特製・極簡版】
    箱の上下端を「平均 ± 標準偏差」にし、最大・最小をヒゲで表現。生データ点は完全排除。
    """
    output_root = Path(output_root)
    # インデックス重複対策
    plot_df_base = merged_summary.copy().reset_index(drop=True)
    targets = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]

    for col in targets:
        if col not in plot_df_base.columns: continue
        
        plt.figure(figsize=(12, 7))
        # 0除外（AIC以外）
        plot_data = plot_df_base[plot_df_base[col] != 0].copy() if col != "AIC" else plot_df_base.copy()
        
        # モデルごとの統計量を算出
        stats = plot_data.groupby("モデル構成")[col].agg(["mean", "std", "min", "max"]).sort_values("mean")
        x_names = stats.index
        x_coords = np.arange(len(x_names))

        # 1. 「平均 ± 標準偏差」の箱を描画
        # 水色の箱が「標準偏差」の広がりを表す
        plt.bar(x_coords, (stats["std"] * 2), bottom=(stats["mean"] - stats["std"]),
                width=0.5, color='skyblue', alpha=0.6, label='平均 ± 1標準偏差', edgecolor='blue', lw=1.5)

        # 2. 平均値に太い赤線を引く
        plt.hlines(stats["mean"], x_coords - 0.25, x_coords + 0.25, color='red', lw=3, label='平均値')

        # 3. 最大・最小を「ヒゲ」として描画
        # これで、最悪のケース（150超えなど）がどこまで伸びたか分かります
        plt.vlines(x_coords, stats["min"], stats["max"], color='black', lw=1.2, ls='-')
        # ヒゲの上下の横棒
        plt.scatter(x_coords, stats["min"], marker='_', color='black', s=200, lw=2)
        plt.scatter(x_coords, stats["max"], marker='_', color='black', s=200, lw=2)

        plt.xticks(x_coords, x_names, rotation=45, ha='right')
        plt.ylabel(col)
        plt.title(f"{col} の統計分布\n[水色箱: 平均±標準偏差, 赤線: 平均, ヒゲ: 最大・最小]")
        plt.grid(axis='y', alpha=0.3, ls=':')
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # 保存
        plt.savefig(output_root / f"全期間_統計箱_{col}.png", dpi=300)
        plt.close()

    print(f"📊 統計箱グラフ（予測RMSE, AICなど全5種）を保存しました。点は消しました。")

def plot_model_boxplots(merged_summary, output_root):
    """
    全指標に対して、箱ひげ図(分布)の上に平均・標準偏差(エラーバー)を重ねて可視化
    """
    output_root = Path(output_root)
    # インデックス重複エラー対策
    plot_df_base = merged_summary.copy().reset_index(drop=True)
    
    # 全指標をループで回す
    targets = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]
    
    for col in targets:
        if col not in plot_df_base.columns:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # 0除外（AIC以外）
        if col != "AIC":
            plot_data = plot_df_base[plot_df_base[col] != 0].copy()
        else:
            plot_data = plot_df_base.copy()
            
        if plot_data.empty: continue

        # 平均値が低い順（優秀な順）に左から並べる
        order = plot_data.groupby("モデル構成")[col].mean().sort_values().index
        
        # 1. 箱ひげ図（分布・中央値・四分位を表示）
        # alphaを下げて、上に重ねるエラーバーを見やすくする
        sns.boxplot(data=plot_data, x="モデル構成", y=col, order=order, 
                    palette="Pastel1", width=0.6, boxprops=dict(alpha=0.4))
        
        # 2. 平均・標準偏差のエラーバー（ご要望の「平均〜標準偏差」の帯）
        # ci="sd" (または errorbar="sd") で標準偏差を表示
        sns.pointplot(data=plot_data, x="モデル構成", y=col, order=order,
                      join=False, color="red", marker="D", scale=0.8, 
                      errorbar="sd", capsize=.15, label="平均 ± 標準偏差")
        
        # 3. 生データ点（実際の全ウィンドウの値をドットで表示）
        sns.stripplot(data=plot_data, x="モデル構成", y=col, order=order, 
                      color="black", alpha=0.2, jitter=True)
        
        plt.xticks(rotation=45, ha='right')
        plt.title(f"モデル構成別 {col} の詳細分布 (全ウィンドウ)\n[箱:中央値, ◆:平均, 赤線:±1標準偏差]", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        # 凡例を表示
        plt.legend(loc='upper left')

        plt.tight_layout()
        
        # 指標名を含めて保存
        save_path = output_root / f"全期間_詳細箱ひげ_{col}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"📊 全指標の箱ひげ図（エラーバー付き）を保存しました。")

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
        plt.title(f"{col} の分布と統計指標 (全ウィンドウ)\n[箱:分布, ◆:平均, 赤縦線:標準偏差]", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_root / f"全期間_ええ感じ箱ひげ_{col}.png", dpi=300)
        plt.close()

def visualize_model_performance(csv_path):
    # CSV（詳細集計版）の読み込み
    df = pd.read_csv(csv_path, index_col=0)
    output_dir = Path(csv_path).parent
    plt.style.use('ggplot')

    # 出力したい指標のリスト
    targets = ["予測RMSE", "AIC", "訓練RMSE", "最大固有値", "RMSE比"]

    for col in targets:
        avg_col = f"{col}_平均"
        std_col = f"{col}_標準偏差"
        
        if avg_col not in df.columns:
            continue

        # --- 1. 棒グラフ ＋ エラーバー（平均と標準偏差） ---
        plt.figure(figsize=(12, 7))
        # 平均値が低い順（良い順）に並べ替え
        df_sorted = df.sort_values(avg_col)
        
        # あなたが「いい感じ」と言ってくれたスタイル
        plt.bar(df_sorted.index, df_sorted[avg_col], 
                yerr=df_sorted[std_col], 
                capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(f"{col} (平均)")
        plt.title(f"モデル構成別 {col} 比較\n(エラーバーは標準偏差を表示)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 指標名をつけて保存
        plt.savefig(output_dir / f"全期間_比較_{col}.png", dpi=300)
        plt.close()

    # --- 2. AIC vs 予測RMSE の散布図（これは1枚だけ作成） ---
    if "AIC_平均" in df.columns and "予測RMSE_平均" in df.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x="AIC_平均", y="予測RMSE_平均", 
                        size="予測RMSE_標準偏差", hue=df.index, 
                        sizes=(100, 1000), alpha=0.6)
        
        for i, txt in enumerate(df.index):
            plt.annotate(txt, (df["AIC_平均"].iloc[i], df["予測RMSE_平均"].iloc[i]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        plt.title("モデルの複雑さ(AIC) vs 予測精度(RMSE)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "全期間_AIC_vs_RMSE_分布.png", dpi=300)
        plt.close()

    print(f"💾 グラフ（棒グラフ5種 ＋ 散布図1種）を {output_dir} に保存しました。")

# =========================================================
# 3. メイン処理
# =========================================================
def main():
    # データ読み込みと前処理（ここは共通）
    df_raw = pd.read_csv(CONFIG["data_path"], index_col=0, parse_dates=True).sort_index()
    sector_df, macro_df, common_idx, meta = prepare_aligned(df_raw)
    
    # --- Moving Window 設定 ---
    test_len = 4
    n_total = len(common_idx)
    
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

        # 1. PCA: 訓練データ(tr_idx_list)でFitし、全期間をTransform
        sc_pca = StandardScaler()
        X_tr_pca = sc_pca.fit_transform(sector_df.iloc[tr_idx_list])
        pca_temp = PCA().fit(X_tr_pca)
        n_pcs = max(3, np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= 0.90) + 1)
        
        pca = PCA(n_components=n_pcs).fit(X_tr_pca)
        pc_cols_orig = [f"PC{k+1}" for k in range(n_pcs)]
        
        # 2. 全期間のPC差分データ生成（リーク防止のためPCAインスタンスは維持）
        # pca_no_leak相当の処理をWindow用に調整
        full_pcs = pca.transform(sc_pca.transform(sector_df))
        pc_diff_all = pd.DataFrame(full_pcs, index=sector_df.index).diff().dropna()
        pc_cols_diff = [f"PC{k+1}_DIFF" for k in range(3)]
        pc_diff_all = pc_diff_all.iloc[:, :3]
        pc_diff_all.columns = pc_cols_diff

        # --- 元の分析レポート(analyze_pca_details)実行 ---
        # 引数はWindowに合わせて調整
        analyze_pca_details(pca, sector_df, pc_cols_orig, root_dir, tr_idx_list[0], tr_idx_list[-1], test_len)
        
        # マクロ同期
        m_df_win = macro_df.loc[pc_diff_all.index]
        combinations = [[]] + [list(c) for r in range(1, 4) for c in itertools.combinations(pc_cols_diff, r)]
        summary_list = []

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
            m_cols, m_dim = list(macro_df.columns), len(macro_df.columns)
            
            # モデル推定
            Y_tr, X_tr = make_design(tr_s[m_cols], tr_s[combo] if combo else None, CONFIG["p_lag"])
            Beta = np.linalg.solve(X_tr.T @ X_tr + CONFIG["ridge"] * np.eye(X_tr.shape[1]), X_tr.T @ Y_tr)
            
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
                        for s in range(1, 13): vals.append(vals[-1] * np.exp(imp_raw[s]) if meth_m == "LOGDIFF" else vals[-1] + imp_raw[s])
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
    # aggregate_results を実行（これでヒートマップと詳細CSVができる）
    aggregate_results(CONFIG["output_dir"])

    # モデル評価の可視化グラフを生成
    output_folder = Path(CONFIG["output_dir"])
    csv_file = output_folder / "全ウィンドウ集計_モデル評価_詳細版.csv"
    
    if csv_file.exists():
        visualize_model_performance(csv_file)
    
    print(f"✅ 全工程完了。画像とCSVを確認してください。")

if __name__ == "__main__":
    main()