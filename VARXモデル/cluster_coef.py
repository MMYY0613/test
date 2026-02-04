import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def cluster_coefficients_save_only(
    coef_csv_path,
    out_dir,
    model_type="PC1_DIFF",
    target_var="GDP",
    k_list=(2, 3, 4, 5, 6),
    use_pca_dim=None,
    random_state=0,
    top_n_heatmap=30,
):
    coef_csv_path = Path(coef_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(coef_csv_path)

    # --- 日付（Window末尾） ---
    df["EndDateStr"] = df["Window"].astype(str).str.split("_").str[-1]
    df["Date"] = pd.to_datetime(df["EndDateStr"], format="%Y%m%d", errors="coerce")

    sub = df[(df["Model_Type"] == model_type) & (df["Target_Variable"] == target_var)].copy()
    sub = sub.dropna(subset=["Date"]).sort_values("Date")
    if sub.empty:
        raise ValueError(f"対象データが空: Model_Type={model_type}, Target={target_var}")

    # --- 係数列だけ抽出 ---
    non_coef = {"Window", "Model_Type", "Target_Variable", "EndDateStr", "Date"}
    coef_cols = [c for c in sub.columns if c not in non_coef]

    X = sub[coef_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")   # 列丸ごとNaNは落とす
    coef_cols = list(X.columns)
    X = X.fillna(0.0)                # 残りNaNは0埋め（簡易）

    # --- 標準化（Windowごとの係数ベクトルを比較するため） ---
    Z = StandardScaler().fit_transform(X.values)  # shape: (n_windows, n_coefs)

    # --- PCA（任意） ---
    if use_pca_dim is not None and use_pca_dim < Z.shape[1]:
        pca = PCA(n_components=use_pca_dim, random_state=random_state)
        Zc = pca.fit_transform(Z)
        evr = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        })
        evr.to_csv(out_dir / f"pca_evr_{model_type}_{target_var}.csv", index=False, encoding="utf-8-sig")
    else:
        Zc = Z

    # =========================================================
    # 1) k選び（silhouette）
    # =========================================================
    score_rows = []
    best_k, best_score = None, -np.inf
    best_labels = None

    for k in k_list:
        if k < 2:
            score_rows.append({"k": k, "silhouette": np.nan})
            continue

        km = KMeans(n_clusters=k, random_state=random_state, n_init=30)
        labels = km.fit_predict(Zc)

        if len(np.unique(labels)) <= 1:
            sc = np.nan
        else:
            sc = float(silhouette_score(Zc, labels))

        score_rows.append({"k": k, "silhouette": sc})

        if not np.isnan(sc) and sc > best_score:
            best_score = sc
            best_k = k
            best_labels = labels

    score_df = pd.DataFrame(score_rows).sort_values("k")
    score_df.to_csv(out_dir / f"silhouette_{model_type}_{target_var}.csv", index=False, encoding="utf-8-sig")

    # silhouette曲線保存
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(score_df["k"], score_df["silhouette"], marker="o", lw=2)
    plt.axhline(0, lw=1, alpha=0.4)
    plt.xticks(score_df["k"].tolist())
    plt.grid(alpha=0.25)
    plt.title(f"Silhouette vs k: {model_type}/{target_var}")
    plt.xlabel("k (#clusters)")
    plt.ylabel("silhouette score")
    plt.tight_layout()
    plt.savefig(out_dir / f"silhouette_curve_{model_type}_{target_var}.png", dpi=200, facecolor="white")
    plt.close()

    # best_k 保険
    if best_k is None:
        best_k = min([k for k in k_list if k >= 2], default=2)
        km = KMeans(n_clusters=best_k, random_state=random_state, n_init=30)
        best_labels = km.fit_predict(Zc)
        best_score = np.nan

    sub["Cluster"] = best_labels

    # =========================================================
    # 2) クラスタ時系列（保存）
    # =========================================================
    plt.figure(figsize=(12, 3))
    plt.scatter(sub["Date"], sub["Cluster"], s=40)
    plt.yticks(sorted(sub["Cluster"].unique()))
    title = (
        f"Cluster over time: {model_type}/{target_var} (k={best_k}, sil={best_score:.3f})"
        if not np.isnan(best_score)
        else f"Cluster over time: {model_type}/{target_var} (k={best_k})"
    )
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"cluster_timeline_{model_type}_{target_var}_k{best_k}.png", dpi=200, facecolor="white")
    plt.close()

    # clusteredデータ保存
    sub.to_csv(out_dir / f"clustered_coeffs_{model_type}_{target_var}_k{best_k}.csv",
               index=False, encoding="utf-8-sig")

    # =========================================================
    # 3) クラスタ別の係数統計（mean/std）
    # =========================================================
    stats = sub.groupby("Cluster")[coef_cols].agg(["mean", "std"])
    stats.columns = [f"{c}__{st}" for c, st in stats.columns]
    stats.to_csv(out_dir / f"cluster_stats_{model_type}_{target_var}_k{best_k}.csv",
                 encoding="utf-8-sig")

    # =========================================================
    # 4) 係数の分離度ランキング（between/within）
    # =========================================================
    overall = sub[coef_cols].mean()
    between = pd.Series(0.0, index=coef_cols)
    within = pd.Series(0.0, index=coef_cols)

    for cl, g in sub.groupby("Cluster"):
        w = len(g) / len(sub)
        mu = g[coef_cols].mean()
        between += w * (mu - overall) ** 2
        within += w * g[coef_cols].var(ddof=0)

    sep_df = pd.DataFrame({
        "between_var": between,
        "within_var": within,
        "separation_score": between / (within + 1e-12),
    }).sort_values("separation_score", ascending=False)

    sep_df.to_csv(out_dir / f"coef_separation_ranking_{model_type}_{target_var}_k{best_k}.csv",
                  encoding="utf-8-sig")

    # =========================================================
    # 5) ヒートマップ（改良：0中心発散色 + 対称スケール）
    #    行=係数, 列=Date
    # =========================================================
    Z_df = pd.DataFrame(Z, index=sub["Date"], columns=coef_cols)

    top_cols = list(sep_df.index[:min(top_n_heatmap, len(sep_df))])
    Z_top = Z_df[top_cols].T

    vmax = float(np.nanmax(np.abs(Z_top.values)))
    vmax = max(vmax, 1e-6)

    fig_w = max(14, 0.9 * len(Z_top.columns))
    fig_h = max(6, 0.32 * len(top_cols))
    plt.figure(figsize=(fig_w, fig_h))

    im = plt.imshow(
        Z_top.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest"
    )

    plt.yticks(range(len(top_cols)), top_cols)

    # x軸ラベル間引き（約12個）
    dates = list(Z_top.columns)
    step = max(1, len(dates) // 12)
    xticks = list(range(0, len(dates), step))
    xlabels = [dates[i].strftime("%Y-%m") for i in xticks]
    plt.xticks(xticks, xlabels, rotation=45, ha="right")

    plt.title(f"Std Coefs Heatmap (top{len(top_cols)} by separation): {model_type}/{target_var} (k={best_k})")
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("standardized beta (z-score)")
    plt.tight_layout()

    plt.savefig(out_dir / f"coef_heatmap_top{len(top_cols)}_{model_type}_{target_var}_k{best_k}.png",
                dpi=200, facecolor="white")
    plt.close()

    # =========================================================
    # 6) 追加：クラスタ順に列（時点）を並べ替えたヒートマップ（レジーム可視化）
    # =========================================================
    tmp = sub[["Date", "Cluster"]].copy()
    tmp = tmp.sort_values(["Cluster", "Date"])
    ordered_dates = list(tmp["Date"])

    Z_sorted = Z_df.loc[ordered_dates, top_cols].T  # 行=coef, 列=Date(クラスタ順)

    vmax2 = float(np.nanmax(np.abs(Z_sorted.values)))
    vmax2 = max(vmax2, 1e-6)

    fig_w = max(14, 0.9 * len(ordered_dates))
    fig_h = max(6, 0.32 * len(top_cols))
    plt.figure(figsize=(fig_w, fig_h))

    im2 = plt.imshow(
        Z_sorted.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax2, vmax=vmax2,
        interpolation="nearest"
    )
    plt.yticks(range(len(top_cols)), top_cols)

    # x軸ラベル間引き
    step = max(1, len(ordered_dates) // 12)
    xticks = list(range(0, len(ordered_dates), step))
    xlabels = [ordered_dates[i].strftime("%Y-%m") for i in xticks]
    plt.xticks(xticks, xlabels, rotation=45, ha="right")

    plt.title(f"Std Coefs Heatmap (cluster-sorted): {model_type}/{target_var} (k={best_k})")
    cbar = plt.colorbar(im2, shrink=0.8)
    cbar.set_label("standardized beta (z-score)")
    plt.tight_layout()

    plt.savefig(out_dir / f"coef_heatmap_cluster_sorted_top{len(top_cols)}_{model_type}_{target_var}_k{best_k}.png",
                dpi=200, facecolor="white")
    plt.close()

    print(f"✅ saved to: {out_dir}")
    print(f"   best_k={best_k}, silhouette={best_score}")
    return best_k


if __name__ == "__main__":
    cluster_coefficients_save_only(
        coef_csv_path="./output_mw_new_tmp_20260205/00_aggregate/coefficients/all_model_coefficients.csv",
        out_dir="./output_mw_new_tmp_20260205/00_aggregate/coefficients/cluster",
        model_type="PC1_DIFF",
        target_var="GDP",
        k_list=(2, 3, 4, 5, 6),
        use_pca_dim=None,
        top_n_heatmap=30,
    )