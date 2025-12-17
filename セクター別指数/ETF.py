import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path

SECTOR_ETF = {
    "FOODS": "1617.JP",
    "ENERGY": "1618.JP",
    "MAT_CHEM": "1620.JP",
    "AUTO": "1622.JP",
    "IT_SERV": "1626.JP",
    "BANKS": "1631.JP",
    "FIN_EX_BANKS": "1632.JP",
    "REAL_ESTATE": "1633.JP",
}
START_DATE = "2010-01-01"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_stooq_close(symbol: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df["Close"].astype(float)

# 日次終値
px = {name: fetch_stooq_close(sym).loc[START_DATE:] for name, sym in SECTOR_ETF.items()}
px = pd.DataFrame(px)

# ---- A) 四半期末（last） ----
px_q_end = px.resample("Q").last()
ret_q_end = np.log(px_q_end).diff()

px_q_end.to_csv(OUT_DIR / "price_q_end.csv", encoding="utf-8-sig")
ret_q_end.to_csv(OUT_DIR / "logret_q_end.csv", encoding="utf-8-sig")

# ---- B) 四半期平均（mean） ----
px_q_mean = px.resample("Q").mean()
ret_q_mean = np.log(px_q_mean).diff()

px_q_mean.to_csv(OUT_DIR / "price_q_mean.csv", encoding="utf-8-sig")
ret_q_mean.to_csv(OUT_DIR / "logret_q_mean.csv", encoding="utf-8-sig")

print("Saved quarterly end + quarterly mean versions.")
print("end:", ret_q_end.tail(3))
print("mean:", ret_q_mean.tail(3))