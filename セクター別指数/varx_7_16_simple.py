import numpy as np
import pandas as pd

DATA_PATH = "data/ret_plus_sent_last8q_7d_mani.csv"
TEST_H = 2
RIDGE = 1e-2   # 7次元なので弱め

RET6 = ["FOODS","ENERGY","MAT_CHEM","AUTO","IT_SERV","BANKS"]

# ---- load
df0 = pd.read_csv(DATA_PATH)
if "Date" in df0.columns:
    df0["Date"] = pd.to_datetime(df0["Date"])
    df0 = df0.set_index("Date")
else:
    df0.index = pd.to_datetime(df0.index)
df0 = df0.sort_index()

# ---- exog: GDP/Nikkei (levelならlog差分)。無ければダミー
EXOG = []
if "GDP_LEVEL" in df0.columns and "GDP_LOGDIFF" not in df0.columns:
    df0["GDP_LOGDIFF"] = np.log(df0["GDP_LEVEL"]).diff()
if "GDP_LOGDIFF" in df0.columns:
    EXOG.append("GDP_LOGDIFF")

if "NIKKEI_LEVEL" in df0.columns and "NIKKEI_LOGRET" not in df0.columns:
    df0["NIKKEI_LOGRET"] = np.log(df0["NIKKEI_LEVEL"]).diff()
if "NIKKEI_LOGRET" in df0.columns:
    EXOG.append("NIKKEI_LOGRET")

if not EXOG:
    rng = np.random.default_rng(42)
    df0["GDP_LOGDIFF"]   = rng.normal(0.002, 0.003, size=len(df0))
    df0["NIKKEI_LOGRET"] = rng.normal(0.01, 0.05, size=len(df0))
    EXOG = ["GDP_LOGDIFF", "NIKKEI_LOGRET"]

df0[EXOG] = df0[EXOG].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---- sentiment average (endogenous)
sent_cols = [f"sent_{c}" for c in RET6]
df0["SENT_AVG"] = df0[sent_cols].mean(axis=1)

ENDOG = RET6 + ["SENT_AVG"]
# RET8 = ["FOODS","ENERGY","MAT_CHEM","AUTO","IT_SERV","BANKS","FIN_EX_BANKS","REAL_ESTATE"]
# ENDOG = RET8 + [f"sent_{c}" for c in RET8]
# RIDGE = 1e0

# ---- build train/test
dfm = df0[ENDOG + EXOG].replace([np.inf, -np.inf], np.nan).dropna()
split = max(len(dfm) - TEST_H, 1)
train = dfm.iloc[:split]
test  = dfm.iloc[split:]

# ---- VARX(1) ridge fit: y_t = c + A1 y_{t-1} + B x_t + e_t
Y = train[ENDOG].values
X = np.concatenate(
    [np.ones((len(train), 1)),
     train[ENDOG].shift(1).values,
     train[EXOG].values],
    axis=1
)

valid = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
Y = Y[valid]
X = X[valid]

k = X.shape[1]
Beta = np.linalg.solve(X.T @ X + RIDGE*np.eye(k), X.T @ Y)

labels = (["const"] + [f"lag1_{c}" for c in ENDOG] + EXOG)
coef = pd.DataFrame(Beta, index=labels, columns=ENDOG)

c  = coef.loc["const"].values
A1 = coef.loc[[f"lag1_{c}" for c in ENDOG]].values.T
B  = coef.loc[EXOG].values.T

# ---- forecast
Y_hist = train[ENDOG].copy()
preds = []
for h in range(len(test)):
    y_prev = Y_hist.iloc[-1].values
    x_now  = test.iloc[h][EXOG].values
    y_hat  = c + A1 @ y_prev + B @ x_now
    preds.append(y_hat)
    Y_hist = pd.concat([Y_hist, pd.DataFrame([y_hat], index=[test.index[h]], columns=ENDOG)])

pred = pd.DataFrame(preds, index=test.index, columns=ENDOG)

# ---- rmse (returns only)
rmse = np.sqrt(((pred[RET6] - test[RET6])**2).mean(axis=0))
print("RMSE (returns6):")
print(rmse.sort_values())

print("\nA1 shape:", A1.shape, "B shape:", B.shape)
print("\nA1 (head):")
print(pd.DataFrame(A1, index=ENDOG, columns=ENDOG).iloc[:5, :5])

print("\nB:")
print(pd.DataFrame(B, index=ENDOG, columns=EXOG))

print("\nPred vs Actual (returns6):")
out = pd.concat([pred[RET6].add_prefix("pred_"), test[RET6].add_prefix("act_")], axis=1)
print(out)