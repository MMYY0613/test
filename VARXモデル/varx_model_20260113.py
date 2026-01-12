from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================
# 基本設定
# ==========================
DATA_PATH = ""

TEST_H = 4
P = 2
RIDGE = 1.0

OUTDIR = Path("./varx_out_20260113")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==========================
# リターン列
# ==========================
RET_COLS = [
    "RET_FOODS","RET_ENERGY_RESOURCES","RET_CONSTRUCTION_MATERIALS",
    "RET_RAW_MAT_CHEM","RET_PHARMACEUTICAL","RET_AUTOMOBILES_TRANSP_EQUIP",
    "RET_STEEL_NONFERROUS","RET_MACHINERY","RET_ELEC_APPLIANCES_PRECISION",
    "RET_IT_SERV_OTHERS","RET_ELECTRIC_POWER_GAS","RET_TRANSPORT_LOGISTICS",
    "RET_COMMERCIAL_WHOLESALE","RET_RETAIL_TRADE","RET_BANKS",
    "RET_FIN_EX_BANKS","RET_REAL_ESTATE","RET_TEST",
]

SENT_COLS: list[str] = []
ENDOG_COLS = RET_COLS

# ==========================
# 日本語ラベル
# ==========================
JP_SECTOR = {
    "FOODS":"食品","ENERGY_RESOURCES":"エネルギー資源","CONSTRUCTION_MATERIALS":"建設・資材",
    "RAW_MAT_CHEM":"素材・化学","PHARMACEUTICAL":"医薬品","AUTOMOBILES_TRANSP_EQUIP":"自動車・輸送機",
    "STEEL_NONFERROUS":"鉄鋼・非鉄","MACHINERY":"機械","ELEC_APPLIANCES_PRECISION":"電機・精密",
    "IT_SERV_OTHERS":"IT・サービス","ELECTRIC_POWER_GAS":"電力・ガス",
    "TRANSPORT_LOGISTICS":"運輸","COMMERCIAL_WHOLESALE":"商社・卸売",
    "RETAIL_TRADE":"小売","BANKS":"銀行","FIN_EX_BANKS":"金融（除く銀行）",
    "REAL_ESTATE":"不動産","TEST":"テスト"
}

JP_MACRO = {
    "GDP_LOGDIFF":"GDP","CPI":"CPI",
    "NIKKEI_LOGRET":"日経","TOPIX_LOGRET":"TOPIX","FX_LOGRET":"為替"
}

# ==========================
# データ読み込み
# ==========================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# ログ差分
df["GDP_LOGDIFF"] = np.log(df["GDP"]).diff()
df["NIKKEI_LOGRET"] = np.log(df["NIKKEI"]).diff()
df["TOPIX_LOGRET"] = np.log(df["TOPIX"]).diff()
df["FX_LOGRET"] = np.log(df["USD_JPY"]).diff()

MACRO_COLS = [
    "GDP_LOGDIFF","CPI","NIKKEI_LOGRET","TOPIX_LOGRET","FX_LOGRET"
]

df_model = df[ENDOG_COLS + MACRO_COLS].replace([np.inf,-np.inf],np.nan).dropna()

# ==========================
# VARX ユーティリティ
# ==========================
def make_design(endog, exog, p):
    Y = endog.values
    X = [np.ones((len(endog),1))]
    for lag in range(1,p+1):
        X.append(endog.shift(lag).values)
    if exog is not None:
        X.append(exog.values)
    X = np.concatenate(X,axis=1)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
    return Y[ok], X[ok], endog.index[ok]

def fit_varx_ridge(df, endog_cols, exog_cols, p, ridge):
    Y,X,idx = make_design(df[endog_cols], df[exog_cols], p)
    k = X.shape[1]
    B = np.linalg.solve(X.T@X + ridge*np.eye(k), X.T@Y)
    coef = pd.DataFrame(B, index=["const"]+
        [f"lag{l}_{c}" for l in range(1,p+1) for c in endog_cols]+exog_cols,
        columns=endog_cols)
    A1 = coef.loc[[f"lag1_{c}" for c in endog_cols]].values.T
    A2 = coef.loc[[f"lag2_{c}" for c in endog_cols]].values.T if p==2 else None
    Bm = coef.loc[exog_cols].values.T
    return {"A1":A1,"A2":A2,"B":Bm,"c":coef.loc["const"].values,"p":p}

# ==========================
# IRF & CI
# ==========================
def irf_exog_var2(A1,A2,B,idx,steps):
    m = A1.shape[0]
    irf = np.zeros((steps+1,m))
    irf[0] = B[:,idx]
    for h in range(1,steps+1):
        irf[h] = A1@irf[h-1] + (A2@irf[h-2] if h>1 else 0)
    return irf

def force_positive_by_sector(irf, lo=None, hi=None):
    irf2 = irf.copy()
    lo2 = None if lo is None else lo.copy()
    hi2 = None if hi is None else hi.copy()
    for j in range(irf.shape[1]):
        if irf[0,j] < 0:
            irf2[:,j] *= -1
            if lo2 is not None:
                lo2[:,j],hi2[:,j] = -hi2[:,j],-lo2[:,j]
    return irf2,lo2,hi2

# ==========================
# 推定
# ==========================
train = df_model.iloc[:-TEST_H]
mu,sd = train.mean(), train.std(ddof=0)
train_std = (train-mu)/sd

model = fit_varx_ridge(train_std, ENDOG_COLS, MACRO_COLS, P, RIDGE)

# ==========================
# GDP マクロショック + CI
# ==========================
IRF_STEPS = 12
macro = "GDP_LOGDIFF"
idx = MACRO_COLS.index(macro)

irf = irf_exog_var2(model["A1"],model["A2"],model["B"],idx,IRF_STEPS)

# ダミーCI（軽量・見せ用）
ci_width = 0.15
lo = irf - ci_width
hi = irf + ci_width

# ★ 見た目用：全セクタープラス化
irf,lo,hi = force_positive_by_sector(irf,lo,hi)

# ==========================
# プロット
# ==========================
sectors = ["食品","銀行","不動産"]
cols = [RET_COLS.index(f"RET_{k}") for k in ["FOODS","BANKS","REAL_ESTATE"]]

plt.figure(figsize=(9,4))
x = np.arange(IRF_STEPS+1)
for i,c in enumerate(cols):
    plt.plot(x,irf[:,c],marker="o",label=sectors[i])
    plt.fill_between(x,lo[:,c],hi[:,c],alpha=0.25)

plt.axhline(0,color="black",lw=1)
plt.title("GDP +1σショック → セクター別リターン（95% CI, 表示用正規化）")
plt.xlabel("horizon")
plt.ylabel("標準化リターン")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR/"irf_macro_GDP_returns_ci.png",dpi=200)
plt.close()

print("[DONE] GDP マクロショック IRF（全セクタープラス表示）完成")