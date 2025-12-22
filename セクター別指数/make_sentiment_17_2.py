import pandas as pd
from pathlib import Path

IN_DIR = Path("data/sentiment_3m")
OUT_PATH = Path("data/sentiment_17_q.csv")

# 17セクター名の並び（列順）
TOPIX17 = [
    "FOODS", "ENERGY_RESOURCES", "CONSTRUCTION_MATERIALS",
    "RAW_MAT_CHEM", "PHARMACEUTICAL",
    "AUTOMOBILES_TRANSP_EQUIP", "STEEL_NONFERROUS",
    "MACHINERY", "ELEC_APPLIANCES_PRECISION",
    "IT_SERV_OTHERS", "ELECTRIC_POWER_GAS",
    "TRANSPORT_LOGISTICS", "COMMERCIAL_WHOLESALE",
    "RETAIL_TRADE", "BANKS", "FIN_EX_BANKS", "REAL_ESTATE",
]

# 33業種 → TOPIX-17 セクター
MAP_33_TO_17 = {
    "水産・農林業": "FOODS",
    "食料品": "FOODS",
    "鉱業": "ENERGY_RESOURCES",
    "石油・石炭製品": "ENERGY_RESOURCES",
    "建設業": "CONSTRUCTION_MATERIALS",
    "ガラス・土石製品": "CONSTRUCTION_MATERIALS",
    "金属製品": "CONSTRUCTION_MATERIALS",
    "化学": "RAW_MAT_CHEM",
    "パルプ・紙": "RAW_MAT_CHEM",
    "繊維製品": "RAW_MAT_CHEM",
    "医薬品": "PHARMACEUTICAL",
    "機械": "MACHINERY",
    "電気機器": "ELEC_APPLIANCES_PRECISION",
    "精密機器": "ELEC_APPLIANCES_PRECISION",
    "ゴム製品": "AUTOMOBILES_TRANSP_EQUIP",
    "輸送用機器": "AUTOMOBILES_TRANSP_EQUIP",
    "鉄鋼": "STEEL_NONFERROUS",
    "非鉄金属": "STEEL_NONFERROUS",
    "情報・通信業": "IT_SERV_OTHERS",
    "サービス業": "IT_SERV_OTHERS",
    "その他製品": "IT_SERV_OTHERS",
    "電気・ガス業": "ELECTRIC_POWER_GAS",
    "陸運業": "TRANSPORT_LOGISTICS",
    "海運業": "TRANSPORT_LOGISTICS",
    "空運業": "TRANSPORT_LOGISTICS",
    "倉庫・運輸関連業": "TRANSPORT_LOGISTICS",
    "卸売業": "COMMERCIAL_WHOLESALE",
    "小売業": "RETAIL_TRADE",
    "銀行業": "BANKS",
    "証券、商品先物取引業": "FIN_EX_BANKS",
    "保険業": "FIN_EX_BANKS",
    "その他金融業": "FIN_EX_BANKS",
    "不動産業": "REAL_ESTATE",
}

records = []

# ① 全ての月次ファイルを読む（3,6,9,12縛りはやめる）
for path in sorted(IN_DIR.glob("*.csv")):
    stem = path.stem          # 例: "201411"
    y, m = int(stem[:4]), int(stem[4:6])

    # 月末日をインデックスにする
    date = pd.to_datetime(stem + "01") + pd.offsets.MonthEnd(0)

    df = pd.read_csv(path)
    # "<all>" だけ使う
    df = df[df["first_keywords"] == "<all>"].copy()

    # 33業種 → 17セクターへマップ
    df["topix17"] = df["sector"].map(MAP_33_TO_17)
    df = df.dropna(subset=["topix17"])

    # セクターごとの平均スコア
    s17 = df.groupby("topix17")["sentiment"].mean()

    rec = {"Date": date}
    for name in TOPIX17:
        rec[name] = s17.get(name, float("nan"))
    records.append(rec)

# ② 月次 DataFrame → 四半期平均に集約
monthly = pd.DataFrame(records).set_index("Date").sort_index()
monthly = monthly[TOPIX17]

# 四半期ごとに平均を取る（2025Q4 は 10・11月だけで平均）
out = monthly.resample("Q").mean()

out.to_csv(OUT_PATH, encoding="utf-8-sig")

print("saved:", OUT_PATH, "shape=", out.shape)
print(out.tail())