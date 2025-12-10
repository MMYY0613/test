import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面なし環境用バックエンド
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

np.random.seed(123)

# =====================================
# 0. 共通設定
# =====================================
T = 80  # 期間
sector_names = ["A", "B", "C"]  # セクター名

# ベースのVAR(1)係数 A0（安定目）
A0 = np.array([
    [0.5,  0.1, 0.05],
    [0.05, 0.6, 0.10],
    [0.02, 0.05, 0.55],
])

# ノイズ共分散
Sigma = np.array([
    [0.0004, 0.0001, 0.00005],
    [0.0001, 0.0005, 0.00010],
    [0.00005,0.00010,0.0003]
])

# ショック時点（中央あたり）
shock_time = T // 2     # 例: 40

# Sent / GDP へのショック（マクロ共通ショック）
shock_sent = -1.5       # sentiment をガツンと下げる
shock_gdp  = -0.8       # gdp も悪化方向へ

# セクター別ショックベクトル（A, B, C への直接ショック）
# 例：Aが一番大きく、B中くらい、C小さめ
sector_shock_vec = np.array([-0.03, -0.02, -0.01])


# =====================================
# 1. Sentiment & GDP の生成
#    ・ベースパス（shockなし）
#    ・ショック入りパス（マクロショック）
# =====================================

eta_sent = np.random.normal(0.0, 0.3, size=T)
eta_gdp  = np.random.normal(0.0, 0.2, size=T)

sent_base = np.zeros(T)
gdp_base  = np.zeros(T)

sent_shock = np.zeros(T)
gdp_shock  = np.zeros(T)

for t in range(1, T):
    # --- ベース（ショックなし） ---
    sent_base[t] = 0.9 * sent_base[t-1] + 0.02 + 0.1 * eta_sent[t]
    gdp_base[t]  = 0.8 * gdp_base[t-1]  + 0.01 + 0.1 * eta_gdp[t]

    # --- ショックあり（センチ & GDP にショックを加える）---
    extra_sent = shock_sent if t == shock_time else 0.0
    extra_gdp  = shock_gdp  if t == shock_time else 0.0

    sent_shock[t] = 0.9 * sent_shock[t-1] + 0.02 + 0.1 * eta_sent[t] + extra_sent
    gdp_shock[t]  = 0.8 * gdp_shock[t-1]  + 0.01 + 0.1 * eta_gdp[t]  + extra_gdp

# 可視化：Sent / GDP のベース vs ショック入り
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

ax[0].plot(sent_base,  label="sent (base)",  alpha=0.8)
ax[0].plot(sent_shock, label="sent (shock)", alpha=0.8)
ax[0].axvline(shock_time, color="red", linestyle="-", linewidth=2)
ax[0].axvspan(shock_time-0.5, shock_time+0.5, color="red", alpha=0.1)
ax[0].set_ylabel("sentiment")
ax[0].legend(loc="upper left")

ax[1].plot(gdp_base,  label="gdp (base)",  alpha=0.8)
ax[1].plot(gdp_shock, label="gdp (shock)", alpha=0.8)
ax[1].axvline(shock_time, color="red", linestyle="-", linewidth=2)
ax[1].axvspan(shock_time-0.5, shock_time+0.5, color="red", alpha=0.1)
ax[1].set_ylabel("gdp")
ax[1].set_xlabel("time")
ax[1].legend(loc="upper left")

fig.suptitle("Sentiment & GDP: base vs shock (shock also affects env)")
plt.tight_layout()
fig.savefig("fig_env_sent_gdp.png", dpi=200)
plt.close(fig)


# =====================================
# 2. センチ & GDP で係数が変わる時変VAR
#
#   A_t = A0 + A_sent * sent_t + A_gdp * gdp_t
#   Y_t = A_t Y_{t-1} + Gamma @ [sent_t, gdp_t] + sector_shock_t + eps_t
#
#   ・Y_base: env = (sent_base, gdp_base), sector_shock=0
#   ・Y_shock: env = (sent_shock, gdp_shock), sector_shock_vec を shock_time に投入
# =====================================

A_sent = np.array([
    [ 0.04, 0.01, 0.00],
    [ 0.00, 0.03, 0.01],
    [-0.01, 0.00, 0.02],
])

A_gdp = np.array([
    [ 0.02, 0.00, 0.01],
    [ 0.01, 0.02, 0.00],
    [ 0.00, 0.01, 0.02],
])

Gamma = np.array([
    [0.05, 0.03],   # A
    [0.03, 0.02],   # B
    [0.04, 0.02],   # C
])

eps_base  = np.random.multivariate_normal(mean=np.zeros(3), cov=Sigma, size=T)
eps_shock = np.random.multivariate_normal(mean=np.zeros(3), cov=Sigma, size=T)

Y_base  = np.zeros((T, 3))
Y_shock = np.zeros((T, 3))

for t in range(1, T):
    # --- ベース用（ショックなし） ---
    S_base = np.array([sent_base[t], gdp_base[t]])
    A_t_base = A0 + A_sent * S_base[0] + A_gdp * S_base[1]
    Y_base[t] = A_t_base @ Y_base[t-1] + Gamma @ S_base + eps_base[t]

    # --- ショック入り（環境＋セクター別ショック） ---
    S_shock = np.array([sent_shock[t], gdp_shock[t]])
    A_t_shock = A0 + A_sent * S_shock[0] + A_gdp * S_shock[1]

    # 各セクターへの直接ショック（t == shock_time のときだけ入る）
    u_t = sector_shock_vec if t == shock_time else np.zeros(3)

    Y_shock[t] = A_t_shock @ Y_shock[t-1] + Gamma @ S_shock + u_t + eps_shock[t]


# =====================================
# 3-A. セクター時系列：ベース vs ショック
# =====================================

fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for i, name in enumerate(sector_names):
    ax[i].plot(Y_base[:, i],  label="env (base)",  alpha=0.9)
    ax[i].plot(Y_shock[:, i], label="env + shocks", alpha=0.9)

    ax[i].axvline(shock_time, color="red", linestyle="-", linewidth=2)
    ax[i].axvspan(shock_time-0.5, shock_time+0.5, color="red", alpha=0.1)
    ax[i].axhline(0, linestyle="--", linewidth=0.8)
    ax[i].set_ylabel(f"Sector {name}")

    if i == 0:
        ax[i].annotate("Shock\n(env + sector A,B,C)",
                       xy=(shock_time, Y_shock[shock_time, i]),
                       xytext=(shock_time+2, Y_shock[shock_time, i] + 0.03),
                       arrowprops=dict(arrowstyle="->", color="red"),
                       color="red")

    ax[i].legend(loc="upper left")

ax[-1].set_xlabel("time")
fig.suptitle("Sector paths: base env vs env + (macro + sector) shocks")
plt.tight_layout()
fig.savefig("fig_sectors_timeseries.png", dpi=200)
plt.close(fig)

# =====================================
# 3-B. ショックの追加インパクト（Y_shock - Y_base）を
#      赤↔青のヒートマップで可視化
# =====================================

delta = Y_shock - Y_base  # (T, 3)

window = 15
t0 = max(shock_time - window, 0)
t1 = min(shock_time + window, T-1)

delta_window = delta[t0:t1+1]
time_axis = np.arange(t0, t1+1)

# カラースケールを「0 を真ん中」に揃える（正負で対称）
vmax = np.max(np.abs(delta_window))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, ax = plt.subplots(figsize=(10, 3.2))

im = ax.imshow(
    delta_window.T,
    aspect="auto",
    origin="lower",
    extent=[time_axis[0], time_axis[-1], -0.5, len(sector_names)-0.5],
    norm=norm,
    cmap="bwr"  # Blue-White-Red：マイナス＝青、プラス＝赤
)

# y 軸：Sector A/B/C
ax.set_yticks(range(len(sector_names)))
ax.set_yticklabels([f"Sector {s}" for s in sector_names])

# x 軸：ショック時点を強調
ax.axvline(shock_time, color="red", linestyle="-", linewidth=2)
ax.axvspan(shock_time-0.5, shock_time+0.5, color="red", alpha=0.08)

ax.set_xlabel("time")
ax.set_title(
    "Shock impact heatmap: Y_shock - Y_base\n"
    "(macro + sector-specific shocks)",
    pad=10
)

# カラーバー
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("difference (shock - base)")

# いちばん効いてるところに × マーカー（任意）
max_idx = np.unravel_index(np.argmax(np.abs(delta_window)), delta_window.shape)
t_star = time_axis[max_idx[0]]
s_star = max_idx[1]
ax.scatter([t_star], [s_star], marker="x", color="black", linewidths=1.2)

# 横方向にだけ“段”が分かるグリッド
ax.set_yticks(
    np.arange(-0.5, len(sector_names), 1),
    minor=True
)
ax.grid(which="minor", axis="y", linewidth=0.4, alpha=0.4)

plt.tight_layout()
fig.savefig("fig_shock_heatmap.png", dpi=200)
plt.close(fig)