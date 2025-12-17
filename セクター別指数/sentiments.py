# pip install gdelt pandas numpy tqdm

import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import gdelt
from pathlib import Path

# ===== 入力 =====
RET_Q_PATH = "data/topix17_proxy_etf_logret_q.csv"   # 四半期末indexのret
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== セクターKW（まずは当たること優先）=====
SECTOR_KW = {
  "BANKS": ["MITSUBISHI UFJ", "MUFG", "SUMITOMO MITSUI", "SMFG", "MIZUHO", "BANK OF JAPAN", "BOJ"],
  "AUTO": ["TOYOTA", "HONDA", "NISSAN", "MAZDA", "SUZUKI", "SUBARU", "DENSO", "EV"],
  "REAL_ESTATE": ["MITSUI FUDOSAN", "MITSUBISHI ESTATE", "SUMITOMO REALTY", "REIT", "REAL ESTATE", "MORTGAGE"],
  "ENERGY": ["INPEX", "ENEOS", "IDEMITSU", "LNG", "CRUDE OIL", "OIL", "GAS", "POWER"],
  "MAT_CHEM": ["NIPPON STEEL", "STEEL", "CHEMICAL", "MITSUBISHI CHEMICAL", "SUMITOMO CHEMICAL", "MATERIALS"],
  "FOODS": ["AJINOMOTO", "KIRIN", "ASAHI", "SUNTORY", "MEIJI", "NISSIN", "FOOD", "BEVERAGE"],
  "IT_SERV": ["NTT", "KDDI", "SOFTBANK", "FUJITSU", "NEC", "SONY", "CLOUD", "DX", "AI", "SOFTWARE"],
  "FIN_EX_BANKS": ["TOKIO MARINE", "MS&AD", "SOMPO", "NOMURA", "DAIWA", "SBI", "INSURANCE", "SECURITIES", "BROKER"],
}

# ===== 設定 =====
WINDOW_DAYS = 7          # ★直近1週間に変更（速い）
START_FROM = "2013-06-30"
N_QUARTERS = 8           # 直近8四半期だけ

# ===== regex（表記揺れ対応）=====
def expand_keywords(words):
    out = set()
    for w in words:
        w2 = w.replace("_", " ")
        out.add(w2)
        out.add(w2.replace(" ", "_"))
        out.add(w2.replace(" ", "-"))
    return list(out)

def compile_regex(words):
    esc = [re.escape(w) for w in expand_keywords(words)]
    return re.compile("|".join(esc), re.IGNORECASE)

regex_by_sector = {sec: compile_regex(words) for sec, words in SECTOR_KW.items()}

# ★全KWユニオン（候補行の事前削減用）
ALL_WORDS = []
for ws in SECTOR_KW.values():
    ALL_WORDS += expand_keywords(ws)
all_regex = re.compile("|".join(map(re.escape, sorted(set(ALL_WORDS)))), re.IGNORECASE)

# ===== tone抽出（この環境は 'TONE' 列がある）=====
def extract_tone(df: pd.DataFrame) -> pd.Series:
    s = df["TONE"].astype(str).str.split(",", expand=True)[0]
    return pd.to_numeric(s, errors="coerce")

# ===== メイン =====
ret_q = pd.read_csv(RET_Q_PATH, index_col=0, parse_dates=True).sort_index()
q_all = ret_q.index[ret_q.index >= pd.Timestamp(START_FROM)]
q_ends = q_all[-N_QUARTERS:]  # 直近8Q

gd = gdelt.gdelt(version=1)

rows = []
debug_done = False

for qend in tqdm(q_ends, desc="quarters"):
    t0 = time.perf_counter()

    start = (qend - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y %b %d")
    end   = qend.strftime("%Y %b %d")
    print(f"\n[FETCH] {qend.date()}  {start} -> {end}")

    try:
        gkg = gd.Search([start, end], table="gkg", output="pandas", coverage=False)
    except Exception as e:
        out = {"date": qend, "note": f"fetch_error: {type(e).__name__}"}
        for sec in SECTOR_KW:
            out[f"sent_{sec}"] = np.nan
            out[f"vol_{sec}"]  = 0
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_DIR / "_sent_partial.csv", index=False, encoding="utf-8-sig")
        continue

    if gkg is None or len(gkg) == 0:
        out = {"date": qend, "note": "empty"}
        for sec in SECTOR_KW:
            out[f"sent_{sec}"] = np.nan
            out[f"vol_{sec}"]  = 0
        rows.append(out)
        pd.DataFrame(rows).to_csv(OUT_DIR / "_sent_partial.csv", index=False, encoding="utf-8-sig")
        continue

    gkg = gkg.copy()
    gkg["tone"] = extract_tone(gkg)

    # ★blobは短く：ORGANIZATIONS + THEMES だけ
    org  = gkg["ORGANIZATIONS"].fillna("").astype(str)
    thm  = gkg["THEMES"].fillna("").astype(str)
    blob = org + " " + thm

    # ★候補行を削る（全KWユニオン一致のみ残す）
    m_all = blob.str.contains(all_regex, na=False)
    gkg2  = gkg.loc[m_all].copy()
    blob2 = blob.loc[m_all]

    if not debug_done:
        print("=== DEBUG (first quarter) ===")
        print("rows fetched:", len(gkg))
        print("rows kept  :", len(gkg2), f"(kept ratio={m_all.mean():.4f})")
        print("TONE raw sample:", gkg["TONE"].head(3).tolist())
        print("tone sample:", gkg["tone"].head(10).tolist())
        print("tone non-null ratio:", float(gkg["tone"].notna().mean()))
        print("contains TOYOTA ratio:", float(blob.str.contains("TOYOTA", case=False, na=False).mean()))
        print("=============================")
        debug_done = True

    out = {"date": qend}
    for sec, r in regex_by_sector.items():
        m = blob2.str.contains(r, na=False)
        sub = gkg2.loc[m, "tone"].dropna()
        out[f"vol_{sec}"]  = int(m.sum())
        out[f"sent_{sec}"] = float(sub.mean()) if len(sub) else np.nan

    rows.append(out)

    # ★途中保存
    pd.DataFrame(rows).to_csv(OUT_DIR / "_sent_partial.csv", index=False, encoding="utf-8-sig")

    t1 = time.perf_counter()
    vols = {k: v for k, v in out.items() if k.startswith("vol_")}
    print(f"[DONE] {qend.date()} elapsed={t1-t0:.1f}s vols={vols}")

sent_q = pd.DataFrame(rows).set_index("date").sort_index()

# ===== 出力 =====
sent_path = OUT_DIR / "sector_sentiment_q_7d_last8q.csv"
sent_q.to_csv(sent_path, encoding="utf-8-sig")

df = ret_q.join(sent_q, how="left")
df_path = OUT_DIR / "ret_plus_sent_last8q_7d.csv"
df.to_csv(df_path, encoding="utf-8-sig")

print("\nSaved:")
print(" ", sent_path)
print(" ", df_path)

print("\nvol sums:")
print(sent_q.filter(like="vol_").sum().sort_values(ascending=False))

print("\nSentiment tail:")
print(sent_q.tail())