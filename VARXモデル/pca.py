import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import matplotlib.dates as mdates

 # 多くのMacにある
mpl.rcParams["axes.unicode_minus"] = False     # マイナス記号の文字化け防止
mpl.rcParams["font.family"] = ["Hiragino Sans"]
os.makedirs("./output_pca", exist_ok=True)

# =========================
# データ読み込み
# =========================
all_q_merge = pd.read_csv("./data/all_q_merged.csv", index_col=0)
all_q_merge.index = pd.to_datetime(all_q_merge.index)
all_q_merge = all_q_merge.sort_index()

SECTOR_COLS = [
    "RET_FOODS","RET_ENERGY_RESOURCES","RET_CONSTRUCTION_MATERIALS",
    "RET_RAW_MAT_CHEM","RET_PHARMACEUTICAL","RET_AUTOMOBILES_TRANSP_EQUIP",
    "RET_STEEL_NONFERROUS","RET_MACHINERY","RET_ELEC_APPLIANCES_PRECISION",
    "RET_IT_SERV_OTHERS","RET_ELECTRIC_POWER_GAS","RET_TRANSPORT_LOGISTICS",
    "RET_COMMERCIAL_WHOLESALE","RET_RETAIL_TRADE","RET_BANKS",
    "RET_FIN_EX_BANKS","RET_REAL_ESTATE","RET_TEST",
]

TEST_SIZE = 4       # 直近TEST_SIZE行をテスト
CUM_TH = 0.90       # 累積寄与率
MAX_PC = 3          # 出力/プロットしたい主成分の本数（Noneなら自動）

JP_LABEL = {
    # "RET_FOODS": "食品",
    # "RET_BANKS": "銀行",
}

def pca_timeseries_no_leak(df: pd.DataFrame,
                           cols: list[str],
                           test_size: int,
                           cum_th: float = 0.9,
                           max_pc: int | None = None):
    """
    時系列PCA（リークなし）
    - 訓練期間で scaler.fit / pca.fit
    - テスト期間は transform のみ
    """
    X = df[cols].copy()

    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna().sort_index()

    n = len(X)
    test_size = int(test_size)
    test_size = max(1, min(test_size, n - 1))
    split = n - test_size

    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    split_ts = X.index[split]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    pca = PCA()
    Ztr_full = pca.fit_transform(Xtr)
    Zte_full = pca.transform(Xte)

    expl = pca.explained_variance_ratio_
    cum = np.cumsum(expl)
    k_auto = int(np.argmax(cum >= cum_th) + 1)

    if max_pc is None:
        k = k_auto
    else:
        k = int(max_pc)
        k = max(1, min(k, len(cols)))

    pc_cols = [f"セクター_PC{i+1}" for i in range(k)]
    pc_train = pd.DataFrame(Ztr_full[:, :k], index=X_train.index, columns=pc_cols)
    pc_test  = pd.DataFrame(Zte_full[:, :k], index=X_test.index,  columns=pc_cols)
    pc_all = pd.concat([pc_train, pc_test], axis=0)

    loading = pd.DataFrame(pca.components_[:k].T, index=cols, columns=pc_cols)
    pca_info = pd.DataFrame({"寄与率": expl[:k], "累積寄与率": cum[:k]}, index=pc_cols)

    return pc_all, loading, pca_info, split_ts, scaler, pca

# 実行
pc_all, loading, pca_info, split_ts, scaler, pca = pca_timeseries_no_leak(
    all_q_merge, SECTOR_COLS, test_size=TEST_SIZE, cum_th=CUM_TH, max_pc=MAX_PC
)

print("=== PCA結果（寄与率・累積寄与率）===")
print(pca_info)

print("\n=== セクター_PC1 の負荷量（絶対値 上位10）===")
top10 = loading[pca_info.index[0]].abs().sort_values(ascending=False).head(10)
top10.index = [JP_LABEL.get(x, x) for x in top10.index]
print(top10)

# =========================
# 保存（CSV）※Excel文字化け防止：utf-8-sig
# =========================
pc_all.to_csv("./output_pca/セクター_主成分スコア.csv", encoding="utf-8-sig")
loading.to_csv("./output_pca/セクター_負荷量.csv", encoding="utf-8-sig")
pca_info.to_csv("./output_pca/主成分_寄与率.csv", encoding="utf-8-sig")

# =========================
# 主成分の線形結合係数（a,b）をCSV出力
# PCk = Σ w_i*(X_i - mu_i)/sigma_i = Σ a_i*X_i + b
# =========================
pc_cols = list(pca_info.index)

a_mat = loading[pc_cols].copy()
a_mat = a_mat.div(scaler.scale_, axis=0)

b_vec = {}
for pc_name in pc_cols:
    w = loading[pc_name].values
    b = -np.sum(w * scaler.mean_ / scaler.scale_)
    b_vec[pc_name] = b
b_ser = pd.Series(b_vec, name="定数項_b")

a_mat.to_csv("./output_pca/主成分_線形結合係数_a.csv", encoding="utf-8-sig")
b_ser.to_csv("./output_pca/主成分_線形結合_定数項_b.csv", encoding="utf-8-sig")

a_with_b = a_mat.copy()
a_with_b.loc["(定数項)"] = b_ser
a_with_b.to_csv("./output_pca/主成分_線形結合係数_a_定数b込み.csv", encoding="utf-8-sig")

print("\n=== 主成分の線形結合係数をCSV出力しました ===")

# =========================
# 図：主成分時系列（年が消えない版）
# =========================
fig, ax = plt.subplots(figsize=(12,4))  # ← constrained_layoutは使わない

pc_all.plot(ax=ax, linewidth=1.6)

ax.axvline(split_ts, linestyle="--", linewidth=1.5)
ax.axhline(0.0, linestyle=":", linewidth=1.2)

ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
ax.grid(True, which="minor", linestyle=":",  linewidth=0.4, alpha=0.3)

# ★ここが重要：x軸の範囲を明示（pandas×datesで事故り防止）
ax.set_xlim(pc_all.index.min(), pc_all.index.max())
ax.margins(x=0)

# x軸：年表示
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))

# ★ラベルを強制表示
ax.tick_params(axis="x", which="both", labelbottom=True)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(0)

ax.set_title(f"セクター主成分スコアの推移（訓練で推定→テストへ適用, テスト={TEST_SIZE}行）")
ax.set_xlabel("日時")
ax.set_ylabel("主成分スコア")

# 凡例：外に出す（tight_layoutで右側スペース確保）
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # ←右を空ける（ここ超大事）

plt.savefig("./output_pca/主成分スコア_時系列.png", dpi=200)
plt.close()

# =========================
# 図：PC1の負荷量（上位10）＋ 0線 ＋ grid
# =========================
pc1_name = pca_info.index[0]
top = loading[pc1_name].abs().sort_values(ascending=False).head(10).index
plot_df = loading.loc[top, [pc1_name]].copy()
plot_df.index = [JP_LABEL.get(x, x) for x in plot_df.index]

plt.figure(figsize=(10,4))
ax = plot_df.plot(kind="bar", ax=plt.gca(), legend=False)
ax.axhline(0.0, linewidth=1.2, linestyle=":")  # 0線

ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

plt.title("セクター_PC1 の負荷量（上位10）")
plt.xlabel("セクター")
plt.ylabel("負荷量（重み）")
plt.tight_layout()
plt.savefig("./output_pca/負荷量_PC1_上位10.png", dpi=200)
plt.close()

print("\n出力先フォルダ: ./output_pca")
print("作成ファイル:")
print("  セクター_主成分スコア.csv / セクター_負荷量.csv / 主成分_寄与率.csv")
print("  主成分_線形結合係数_a.csv / 主成分_線形結合_定数項_b.csv / 主成分_線形結合係数_a_定数b込み.csv")
print("  主成分スコア_時系列.png / 負荷量_PC1_上位10.png")

PLOT_PC = min(5, pc_all.shape[1])   # 上位5まで
Z = pc_all.iloc[:, :PLOT_PC].T.values

plt.figure(figsize=(12, 3.2))
ax = plt.gca()
im = ax.imshow(Z, aspect="auto", interpolation="nearest")

ax.axvline(np.where(pc_all.index == split_ts)[0][0], linestyle="--", linewidth=1.5)
ax.set_yticks(range(PLOT_PC))
ax.set_yticklabels(pc_all.columns[:PLOT_PC])
ax.set_title(f"セクター主成分スコア（ヒートマップ, 上位{PLOT_PC}）")
ax.set_xlabel("時間（左→右）")
ax.set_ylabel("主成分")

plt.colorbar(im, ax=ax, shrink=0.8, label="スコア")
plt.tight_layout()
plt.savefig("./output_pca/主成分スコア_ヒートマップ.png", dpi=200)
plt.close()

plt.figure(figsize=(6,4))
ax = plt.gca()

ax.step(range(1, len(pca_info)+1), pca_info["累積寄与率"].values, where="mid")
ax.scatter(range(1, len(pca_info)+1), pca_info["累積寄与率"].values)

ax.axhline(CUM_TH, linestyle="--", linewidth=1.2)
ax.set_ylim(0, 1.02)
ax.set_xlabel("主成分数")
ax.set_ylabel("累積寄与率")
ax.set_title("累積寄与率（どこまで採用するか）")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("./output_pca/累積寄与率.png", dpi=200)
plt.close()

pc = pc_all.columns[0]  # PC1
is_test = pc_all.index >= split_ts

plt.figure(figsize=(7,4))
ax = plt.gca()
ax.hist(pc_all.loc[~is_test, pc], bins=25, alpha=0.7, label="train")
ax.hist(pc_all.loc[is_test, pc],  bins=25, alpha=0.7, label="test")

ax.set_title(f"{pc} の分布（train vs test）")
ax.set_xlabel("スコア")
ax.set_ylabel("頻度")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

plt.tight_layout()
plt.savefig("./output_pca/PC1_分布_train_test.png", dpi=200)
plt.close()
