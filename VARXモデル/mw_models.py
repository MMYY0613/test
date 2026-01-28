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
from scipy.optimize import lsq_linear  # 追加

warnings.simplefilter('ignore')

# =========================================================
# 1. 設定・定数
# =========================================================
CONFIG = {
    "data_path": "./data/all_q_merged_tmp2.csv",
    "output_dir": "./output_pca_final_tmp2",
    "train_range": ("2015-01-01", "2016-06-30"),
    "test_steps": 4,
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

    # セクター群の定義 (例: あなたが注目したいA, B, C)
    sector_groups = {
        "PC1": SECTOR_COLS, # PC1は全セクター平均で正（景気全体）
        "PC2": ["RET_FOODS", "RET_RETAIL_TRADE", "RET_PHARMACEUTICAL"], # 内需系
        "PC3": ["RET_MACHINERY", "RET_ELEC_APPLIANCES_PRECISION", "RET_STEEL_NONFERROUS"] # 輸出系
    }

    # 設定：上位何件のセクターで符号を判定するか
    top_n_for_sign = 5 

    for i, pc_name in enumerate(pc_cols):
        if pc_name in sector_groups:
            target_sectors = sector_groups[pc_name]
            # 指定されたグループに属するセクターの、現在の負荷量を取得
            current_loadings = pd.Series(components[i], index=sector_df.columns)
            group_loadings = current_loadings[current_loadings.index.isin(target_sectors)]
            
            if not group_loadings.empty:
                # 【ここがポイント】絶対値が大きい上位 n 件のみを抽出
                top_loadings = group_loadings.abs().sort_values(ascending=False).head(top_n_for_sign)
                # 元の符号付きの値を参照して平均を計算
                sign_check_value = group_loadings[top_loadings.index].mean()
                
                # その群の（有力なセクターの）平均が負なら、符号を反転
                if sign_check_value < 0:
                    components[i] *= -1
                    score_df.iloc[:, i] *= -1
                # これで、このPCが上がれば指定セクターも上がるという関係が固定される
        # print(f"  [PCA Fix] {pc_name} sign check finished.")
    
    # 符号の反転処理（解釈を容易にするため）
    # for i in range(components.shape[0]):
    #     if np.mean(components[i]) < 0:
    #         components[i] *= -1
    #         score_df.iloc[:, i] *= -1 

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
        # plt.savefig(output_dir / f"全期間_比較_{col}.png", dpi=300)
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
# 4. 全ウィンドウ連結予測グラフの作成
# =========================================================
def plot_connected_forecasts(output_root, df_raw, meta):
    import matplotlib.style
    # 背景を白に固定し、グラフの体裁を整える
    plt.style.use('default') 
    plt.rcParams['font.family'] = ["Hiragino Sans"]
    
    output_root = Path(output_root)
    m_cols_transformed = list(meta.keys())
    
    # 4つの開始パターンごとにデータを格納
    all_preds = {m: {p: [] for p in range(4)} for m in m_cols_transformed}
    rmse_list = {m: {p: [] for p in range(4)} for m in m_cols_transformed}

    window_dirs = sorted(output_root.glob("Window_Test_*"))
    
    for i, window_path in enumerate(window_dirs):
        phase = i % 4
        
        # 探索範囲を広げる：すべてのサブディレクトリからその変数のCSVを探す
        for sub in window_path.iterdir():
            if not sub.is_dir(): continue
            
            for m_key, m_info in meta.items():
                orig_name = m_info["orig"]
                # 「予測値_水準ベース_変数名.csv」をピンポイントで探す
                target_csv = sub / f"予測値_水準ベース_{orig_name}.csv"
                
                if target_csv.exists():
                    pdf = pd.read_csv(target_csv, index_col=0, parse_dates=True)
                    col_name = f"{orig_name}_Pred"
                    if col_name in pdf.columns:
                        preds = pdf[col_name]
                        all_preds[m_key][phase].append(preds)
                        # RMSE計算
                        actual = df_raw.loc[preds.index, orig_name]
                        if not actual.dropna().empty:
                            rmse = np.sqrt(np.mean((actual - preds)**2))
                            rmse_list[m_key][phase].append(rmse)

    # グラフ作成
    for m_col, m_info in meta.items():
        orig_name = m_info["orig"]
        if not any(all_preds[m_col].values()): continue # データが一つもなければスキップ

        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        ax.set_facecolor('white') # グラフエリアも白
        
        # 実績値（実線・黒）
        ax.plot(df_raw.index, df_raw[orig_name], color='black', lw=2, label="実績値", zorder=2)
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        rmse_summary = []
        
        for p in range(4):
            if not all_preds[m_col][p]: continue
            
            combined = pd.concat(all_preds[m_col][p]).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            
            avg_rmse = np.mean(rmse_list[m_col][p]) if rmse_list[m_col][p] else 0
            rmse_summary.append(f"P{p+1}:{avg_rmse:.3f}")
            
            # 予測値（点線）
            ax.plot(combined.index, combined.values, color=colors[p], linestyle='--', 
                    marker='o', markersize=4, alpha=0.9, 
                    label=f"パターン{p+1} (RMSE: {avg_rmse:.3f})", zorder=3)

        # 凡例とタイトルの設定
        total_avg = np.mean([np.mean(v) for v in rmse_list[m_col].values() if v])
        ax.set_title(f"全期間連結予測: {orig_name}\n【RMSE】{' / '.join(rmse_summary)} (平均: {total_avg:.3f})", 
                     fontsize=14, pad=20)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white')
        ax.grid(True, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel("水準 (Level)")
        ax.set_xlabel("年月")
        
        # X軸のフォーマット
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        plt.savefig(output_root / f"全期間連結予測_{orig_name}.png", dpi=200, facecolor='white')
        plt.close()

import matplotlib.dates as mdates

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
    pc_cols = [c for c in sub_df.columns if "PC" in c]

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
    df_raw = pd.read_csv(CONFIG["data_path"], index_col=0, parse_dates=True).sort_index()
    sector_df, macro_df, common_idx, meta = prepare_aligned(df_raw)
    
    # --- Moving Window 設定 ---
    test_len = 4
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
            # --- 修正版：Ridgeと符号制約を同時に適用するロジック ---
            # --- 修正版：係数範囲の厳格化とCSV保存 (エラー解消版) ---

            # 1. すべての制約をここに集約 (範囲指定 or 符号指定)
            STRICT_CONSTRAINTS = {
                ("PC1", "GDP"): (0.3, 0.7),
                ("PC1", "NIKKEI"): (0.3, 0.7),
                ("PC1", "CPI"): (0.1, 0.5),
                ("PC1", "UNEMP_RATE"): (-0.7, -0.3),
                # 以前の BASE_SIGN_CONSTRAINTS 相当もここに書いておけばエラーになりません
                ("*", "UNEMP"): (-np.inf, -0.3), # 必要なら追加
            }

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
                    beta_list.append(res.x)

                    # CSV出力用のレコード作成
                    # パラメータ名は [CONST, LAG_GDP, ..., PC1_DIFF, ...] の順
                    param_names = ["CONST"] + [f"LAG_{m}" for m in m_cols] + list(combo)

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

            # --- 予測結果の可視化 (日本語化 & i=0 スキップ版) ---
            
            # --- 予測結果の可視化 (逐次予測・日付固定・日本語版) ---

            # i=0 (最初期) を除外し、かつ予測値が存在する場合のみ実行
            if y_te_pred_scaled is not None and i != 0:
                
                # 1. 予測期間の日付インデックスを取得
                start_d_idx = np.where(common_idx == te_dates[0])[0][0]
                te_actual_dates = common_idx[start_d_idx : start_d_idx + CONFIG["test_steps"]]
                
                # 2. 表示範囲 (過去の実績も見えるように前後を調整)
                plot_start_idx = max(0, start_d_idx - 8) 
                plot_end_idx = min(len(common_idx), start_d_idx + CONFIG["test_steps"] + 4)
                full_display_range = common_idx[plot_start_idx:plot_end_idx]

                for i_m, m_col in enumerate(m_cols):
                    orig_name = meta[m_col]["orig"]
                    meth = meta[m_col]["method"]
                    
                    # 3. 水準復元ロジック
                    # スケーリング解除
                    pred_change = (y_te_pred_scaled[:, i_m] * stds[m_col]) + means[m_col]
                    # i != 0 なので必ず直前の実績(ラグ)が取得可能
                    last_actual_level = df_raw.loc[:te_actual_dates[0], orig_name].iloc[-2]
                    
                    pred_levels = []
                    curr_level = last_actual_level
                    for val in pred_change:
                        if meth == "LOGDIFF":
                            curr_level = curr_level * np.exp(val)
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
                    plt.plot(full_display_range, df_raw.loc[full_display_range, orig_name], 
                             label="実績値", color="#333333", marker='o', markersize=5, alpha=0.8)
                    plt.plot(te_actual_dates, pred_levels, 
                             label="予測値 (逐次)", color="red", linestyle="--", marker="x", markersize=7, lw=2)
                    
                    # テスト期間の背景
                    plt.axvspan(te_actual_dates[0], te_actual_dates[-1], color='gray', alpha=0.1, label='予測対象期間')
                    
                    # --- 5. 日付目盛りの固定 (3, 6, 9, 12月のみ表示) ---
                    ax = plt.gca()
                    # 四半期末に目盛りを強制
                    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[3, 6, 9, 12]))
                    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y年%m月'))
                    
                    plt.title(f"【予測】{orig_name} : {m_name_display}", fontsize=14)
                    plt.xlabel("年月", fontsize=12)
                    plt.ylabel("水準 (Level)", fontsize=12)
                    plt.legend(loc='best', frameon=True, shadow=True)
                    plt.grid(True, which='major', linestyle='--', alpha=0.4)
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
    # 1. 全ウィンドウの統計・ヒートマップ集計
    aggregate_results(CONFIG["output_dir"])

    output_folder = Path(CONFIG["output_dir"])
    csv_file = output_folder / "全ウィンドウ集計_モデル評価_詳細版.csv"
    
    # 2. 箱ひげ図や棒グラフなどの評価可視化
    if csv_file.exists():
        visualize_model_performance(csv_file)
    
    # 3. ★ここで実行：全期間連結予測グラフの生成
    print(f"📊 全期間の連結予測グラフを生成中...")
    plot_connected_forecasts(CONFIG["output_dir"], df_raw, meta)

    # --- [修正箇所] main() の最後 ---
    if all_coefficients_records:
        df_coef = pd.DataFrame(all_coefficients_records)
        # 係数が 0.3 や 0.7 に張り付いているか確認しやすくするため
        output_path = Path(CONFIG["output_dir"]) / "all_model_coefficients.csv"
        df_coef.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 全ウィンドウの係数詳細を保存しました: {output_path}")
    
    print(f"✅ 全工程完了。{CONFIG['output_dir']} 内の画像とCSVを確認してください。")

    # 係数行列
    # mainの中での呼び出し例
    output_path = Path(CONFIG["output_dir"]) / "all_model_coefficients.csv"
    for t in ["GDP", "NIKKEI", "USD_JPY"]:
        plot_exog_trends(output_path, CONFIG["output_dir"], target_var=t)

if __name__ == "__main__":
    main()