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

warnings.simplefilter("ignore")

# =========================================================
# 1. 設定・定数（※ここは触らずそのまま）
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

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["Hiragino Sans"]


# =========================================================
# 2. 前処理（※結果同一：中身は同じ処理）
# =========================================================
def smart_transform(series: pd.Series, name: str):
    if (series <= 0).any() or ("JGB" in name) or ("UNEMP" in name):
        return series.diff(), "DIFF"
    return np.log(series).diff(), "LOGDIFF"


def prepare_aligned(df_raw: pd.DataFrame):
    # macro: 変換して整形
    m_work = pd.DataFrame(index=df_raw.index)
    meta = {}
    for col in TARGET_MACRO:
        if col in df_raw.columns:
            ts, method = smart_transform(df_raw[col], col)
            m_work[f"{col}_{method}"] = ts
            meta[f"{col}_{method}"] = {"orig": col, "method": method}
    m_df = m_work.dropna()

    # sector: そのままの仕様（interpolate + diff/dropnaの条件）
    s_raw = df_raw[[c for c in SECTOR_COLS if c in df_raw.columns]].interpolate(
        limit_direction="both"
    )
    s_diff = (
        s_raw.dropna()
        if all(c.startswith("RET_") for c in s_raw.columns)
        else s_raw.diff().dropna()
    )

    # common index
    common = s_diff.index.intersection(m_df.index)
    return s_diff.loc[common], m_df.loc[common], common, meta


# =========================================================
# 3. 基本計算関数（※使ってるロジックは変えない）
# =========================================================
def make_design(endog: pd.DataFrame, exog: pd.DataFrame | None, p: int):
    Y = endog.values
    X_parts = [np.ones((len(endog), 1))]
    for lag in range(1, p + 1):
        X_parts.append(endog.shift(lag).values)
    if exog is not None and not exog.empty:
        X_parts.append(exog.values)

    X = np.concatenate(X_parts, axis=1)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid]


# =========================================================
# 4. PCAレポート（※保存ファイル名・内容維持）
# =========================================================
def analyze_pca_details(
    pca: PCA,
    sector_df: pd.DataFrame,
    pc_cols: list[str],
    root_dir: Path,
    t_start: int,
    t_end: int,
    t_size: int,
):
    """PCA分析：標準化データからの生スコア算出・プロット・負荷量CSV出力"""

    pca_dir = root_dir / "pca_analysis"
    pca_dir.mkdir(exist_ok=True)

    # 訓練+テスト期間のデータを抽出して標準化
    X_target = sector_df.iloc[t_start : t_end + t_size]
    sc = StandardScaler()
    sc.fit(sector_df.iloc[t_start:t_end])
    X_scaled = sc.transform(X_target)

    # 主成分スコア
    scores = pca.transform(X_scaled)
    score_df = pd.DataFrame(scores, index=X_target.index, columns=pc_cols)
    score_df.to_csv(pca_dir / "主成分スコア_生データ.csv", encoding="utf-8-sig")

    expl = pca.explained_variance_ratio_
    components = pca.components_.copy()

    # 符号反転（元ロジック）
    for i in range(components.shape[0]):
        if np.mean(components[i]) < 0:
            components[i] *= -1
            score_df.iloc[:, i] *= -1

    # 負荷量
    loadings = pd.DataFrame(components.T, index=sector_df.columns, columns=pc_cols)
    loadings.to_csv(pca_dir / "セクターの負荷量_一覧.csv", encoding="utf-8-sig")

    # 寄与率
    expl_df = pd.DataFrame(
        expl.reshape(1, -1), columns=pc_cols, index=["ExplainedVariance"]
    )
    expl_df.to_csv(pca_dir / "主成分_寄与率.csv", encoding="utf-8-sig")

    # スコア推移（PC3まで）
    plt.figure(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, pc in enumerate(pc_cols[:3]):
        plt.plot(
            score_df.index,
            score_df[pc],
            label=f"{pc} (寄与:{expl[i]*100:.1f}%)",
            color=colors[i],
            lw=2,
        )
    plt.title("主成分スコアの推移 (セクター標準化後の生値)")
    plt.xlabel("日付")
    plt.ylabel("主成分スコア")
    plt.axhline(0, color="black", lw=1)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(pca_dir / "PCA_スコア推移_生値.png")
    plt.close()

    # 負荷量TOP10（PC3まで）
    for i, pc in enumerate(pc_cols[:3]):
        plt.figure(figsize=(10, 5))
        loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(
            10
        ).plot(kind="bar", color=colors[i])
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


# =========================================================
# 5. 集計・可視化（※保存名・ロジック維持）
# =========================================================
def plot_abs_heatmaps(full_pca_df: pd.DataFrame, output_root: Path):
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
        plt.title(
            f"{pc} 構成セクターの負荷量推移（絶対値）\n[平均寄与率: {mean_var:.1f}%]",
            fontsize=14,
        )
        plt.xlabel("テストウィンドウ (Window)", fontsize=12)
        plt.ylabel("セクター", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_root / f"全期間_{pc}_構造推移_絶対値.png")
        plt.close()


def plot_final_boxplots(merged_summary: pd.DataFrame, output_root: Path):
    plot_df_base = merged_summary.copy().reset_index(drop=True)
    targets = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]

    for col in targets:
        if col not in plot_df_base.columns:
            continue

        plt.figure(figsize=(14, 8))
        plot_data = (
            plot_df_base[plot_df_base[col] != 0].copy()
            if col != "AIC"
            else plot_df_base.copy()
        )
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
        plt.title(f"{col} の分布と統計指標 (全ウィンドウ)\n[箱:分布, ◆:平均, 赤縦線:標準偏差]", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(output_root / f"全期間_ええ感じ箱ひげ_{col}.png", dpi=300)
        plt.close()


def visualize_model_performance(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)
    output_dir = Path(csv_path).parent
    plt.style.use("ggplot")

    targets = ["予測RMSE", "AIC", "訓練RMSE", "最大固有値", "RMSE比"]

    for col in targets:
        avg_col = f"{col}_平均"
        std_col = f"{col}_標準偏差"
        if avg_col not in df.columns:
            continue

        plt.figure(figsize=(12, 7))
        df_sorted = df.sort_values(avg_col)

        plt.bar(
            df_sorted.index,
            df_sorted[avg_col],
            yerr=df_sorted[std_col],
            capsize=5,
            color="skyblue",
            edgecolor="navy",
            alpha=0.7,
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f"{col} (平均)")
        plt.title(f"モデル構成別 {col} 比較\n(エラーバーは標準偏差を表示)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"全期間_比較_{col}.png", dpi=300)
        plt.close()

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

    print(f"💾 グラフ（棒グラフ5種 ＋ 散布図1種）を {output_dir} に保存しました。")


def aggregate_results(output_root: str | Path):
    output_root = Path(output_root)
    all_summaries = []
    pca_abs_details = []

    for window_path in sorted(output_root.glob("Window_Test_*")):
        window_name = window_path.name.replace("Window_Test_", "")

        # モデル比較サマリー
        summary_file = window_path / "モデル比較サマリー.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df["Test_Window"] = window_name
            all_summaries.append(df)

        # PCA負荷量（絶対値）& 寄与率
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

    # ええ感じ箱ひげ（同じ見た目）
    plot_final_boxplots(merged_summary, output_root)

    # 統計詳細CSV
    target_cols = ["訓練RMSE", "予測RMSE", "RMSE比", "AIC", "最大固有値"]
    stats_list = []
    for col in target_cols:
        if col not in merged_summary.columns:
            continue

        temp = merged_summary.copy()
        if col != "AIC":
            temp[col] = temp[col].replace(0, np.nan)

        res = temp.groupby("モデル構成")[col].agg(
            [
                (f"{col}_平均", "mean"),
                (f"{col}_標準偏差", "std"),
                (f"{col}_最大", "max"),
                (f"{col}_最小", "min"),
            ]
        )
        stats_list.append(res)

    model_perf_detail = pd.concat(stats_list, axis=1)
    model_perf_detail["有効試行回数"] = merged_summary.groupby("モデル構成")["予測RMSE"].apply(
        lambda x: (x != 0).sum()
    )
    model_perf_detail = model_perf_detail.round(4)
    model_perf_detail.to_csv(
        output_root / "全ウィンドウ集計_モデル評価_詳細版.csv", encoding="utf-8-sig"
    )

    # PCA構造
    if pca_abs_details:
        full_pca_df = pd.concat(pca_abs_details)
        full_pca_df.to_csv(
            output_root / "全ウィンドウ集計_PCA構造_絶対値詳細.csv",
            encoding="utf-8-sig",
        )
        plot_abs_heatmaps(full_pca_df, output_root)

    print("✅ すべての集計・画像保存（ヒートマップ、箱ひげ図、統計詳細）が完了しました。")


# =========================================================
# 6. メイン（※処理順・保存物・計算は同一）
# =========================================================
def main():
    df_raw = pd.read_csv(CONFIG["data_path"], index_col=0, parse_dates=True).sort_index()
    sector_df, macro_df, common_idx, meta = prepare_aligned(df_raw)

    test_len = 4
    n_total = len(common_idx)

    for i in range(n_total - test_len + 1):
        te_idx_list = np.arange(i, i + test_len)
        tr_idx_list = np.setdiff1d(np.arange(n_total), te_idx_list)

        d_start = common_idx[te_idx_list[0]]
        d_end = common_idx[te_idx_list[-1]]

        root_dir = Path(CONFIG["output_dir"]) / f"Window_Test_{d_start:%Y%m%d}_{d_end:%Y%m%d}"
        root_dir.mkdir(parents=True, exist_ok=True)

        print(f"▶️ 実行中: {root_dir.name}")

        # PCA: 訓練(tr_idx_list)でfitし、全期間transform
        sc_pca = StandardScaler()
        X_tr_pca = sc_pca.fit_transform(sector_df.iloc[tr_idx_list])
        pca_temp = PCA().fit(X_tr_pca)
        n_pcs = max(3, np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= 0.90) + 1)

        pca = PCA(n_components=n_pcs).fit(X_tr_pca)
        pc_cols_orig = [f"PC{k+1}" for k in range(n_pcs)]

        # 全期間PC（差分化、先頭dropna、PC1-3のみ）
        full_pcs = pca.transform(sc_pca.transform(sector_df))
        pc_diff_all = pd.DataFrame(full_pcs, index=sector_df.index).diff().dropna()
        pc_diff_all = pc_diff_all.iloc[:, :3]
        pc_cols_diff = [f"PC{k+1}_DIFF" for k in range(3)]
        pc_diff_all.columns = pc_cols_diff

        # PCAレポート
        analyze_pca_details(
            pca,
            sector_df,
            pc_cols_orig,
            root_dir,
            tr_idx_list[0],
            tr_idx_list[-1],
            test_len,
        )

        # 同期
        m_df_win = macro_df.loc[pc_diff_all.index]
        combinations = [[]] + [
            list(c) for r in range(1, 4) for c in itertools.combinations(pc_cols_diff, r)
        ]

        summary_list = []
        m_cols = list(macro_df.columns)
        m_dim = len(m_cols)

        for combo in combinations:
            m_name_display = ", ".join(combo) if combo else "マクロのみ"
            sub_dir = root_dir / ("_".join(combo) if combo else "BASE_VAR_ONLY")
            sub_dir.mkdir(parents=True, exist_ok=True)

            win_idx = m_df_win.index

            tr_pre_dates = common_idx[tr_idx_list[tr_idx_list < te_idx_list[0]]]
            tr_post_dates = common_idx[tr_idx_list[tr_idx_list > te_idx_list[-1]]]
            te_dates = common_idx[te_idx_list]

            # 過去側訓練
            tr_pre = pd.concat(
                [
                    m_df_win.loc[win_idx.isin(tr_pre_dates)],
                    pc_diff_all.loc[win_idx.isin(tr_pre_dates), combo],
                ],
                axis=1,
            )

            # 未来側訓練
            tr_post = pd.concat(
                [
                    m_df_win.loc[win_idx.isin(tr_post_dates)],
                    pc_diff_all.loc[win_idx.isin(tr_post_dates), combo],
                ],
                axis=1,
            )

            # 元ロジック：未来側先頭をNaNにして「壁」を作る
            if not tr_post.empty:
                tr_post.iloc[0, :] = np.nan

            tr_raw = pd.concat([tr_pre, tr_post])
            te_raw = pd.concat(
                [
                    m_df_win.loc[win_idx.isin(te_dates)],
                    pc_diff_all.loc[win_idx.isin(te_dates), combo],
                ],
                axis=1,
            )

            means = tr_raw.mean()
            stds = tr_raw.std(ddof=0).replace(0, 1.0)

            tr_s = (tr_raw - means) / stds
            te_s = (te_raw - means) / stds

            # 推定
            Y_tr, X_tr = make_design(tr_s[m_cols], tr_s[combo] if combo else None, CONFIG["p_lag"])
            Beta = np.linalg.solve(
                X_tr.T @ X_tr + CONFIG["ridge"] * np.eye(X_tr.shape[1]),
                X_tr.T @ Y_tr,
            )

            # 指標
            tr_resid = Y_tr - X_tr @ Beta
            n_obs = len(Y_tr)
            tr_rmse = np.sqrt(np.mean(tr_resid ** 2))

            Y_te, X_te = make_design(te_s[m_cols], te_s[combo] if combo else None, CONFIG["p_lag"])
            te_rmse = np.sqrt(np.mean((Y_te - X_te @ Beta) ** 2)) if len(Y_te) > 0 else np.nan

            sigma_matrix = (tr_resid.T @ tr_resid) / n_obs
            sign, logdet = np.linalg.slogdet(sigma_matrix)
            if sign > 0:
                aic_val = n_obs * logdet + 2 * Beta.size
            else:
                eigs = np.linalg.eigvalsh(sigma_matrix)
                valid_eigs = eigs[eigs > 1e-15]
                aic_val = n_obs * np.sum(np.log(valid_eigs)) + 2 * Beta.size if len(valid_eigs) > 0 else np.nan

            A1 = Beta[1 : 1 + m_dim, :].T
            max_eig_abs = np.max(np.abs(np.linalg.eigvals(A1)))

            summary_list.append(
                {
                    "モデル構成": m_name_display,
                    "訓練RMSE": round(tr_rmse, 4),
                    "予測RMSE": round(te_rmse, 4) if not np.isnan(te_rmse) else np.nan,
                    "RMSE比": round(te_rmse / tr_rmse, 2)
                    if tr_rmse > 0 and not np.isnan(te_rmse)
                    else np.nan,
                    "AIC": round(aic_val, 2) if not np.isnan(aic_val) else np.nan,
                    "最大固有値": round(max_eig_abs, 3),
                }
            )

            # IRF（元のまま）
            if CONFIG["do_irf"] and combo:
                B = Beta[1 + m_dim :, :].T
                for j, pc_label in enumerate(combo):
                    pc_folder = sub_dir / f"Shock_{pc_label}"
                    pc_folder.mkdir(parents=True, exist_ok=True)

                    impact = np.zeros((13, m_dim))
                    impact[1] = B[:, j]
                    for h in range(2, 13):
                        impact[h] = A1 @ impact[h - 1]

                    for i_m, m_col in enumerate(m_cols):
                        orig_m, meth_m = meta[m_col]["orig"], meta[m_col]["method"]
                        base = BASE_LEVELS.get(orig_m, 100)
                        imp_raw = impact[:, i_m] * stds[m_col]

                        vals = [base]
                        for s in range(1, 13):
                            vals.append(
                                vals[-1] * np.exp(imp_raw[s]) if meth_m == "LOGDIFF" else vals[-1] + imp_raw[s]
                            )

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

        pd.DataFrame(summary_list).sort_values("AIC").to_csv(
            root_dir / "モデル比較サマリー.csv", index=False, encoding="utf-8-sig"
        )

    # 全体集計
    aggregate_results(CONFIG["output_dir"])

    # 追加可視化
    output_folder = Path(CONFIG["output_dir"])
    csv_file = output_folder / "全ウィンドウ集計_モデル評価_詳細版.csv"
    if csv_file.exists():
        visualize_model_performance(csv_file)

    print("✅ 全工程完了。画像とCSVを確認してください。")


if __name__ == "__main__":
    main()