python - <<'PY'
import numpy as np, pandas as pd

DATA_PATH = "./data/all_q_merged.csv"
test_size = 4
p = 1
min_train = 12

df = pd.read_csv(DATA_PATH, index_col=0)
idx = pd.to_datetime(df.index, errors="coerce")
idx = idx[~idx.isna()]
idx = idx.sort_values()
idx = pd.DatetimeIndex(idx.unique())

n = len(idx)

with open("splits.tsv","w",encoding="utf-8") as f:
    for start_pos in range(0, n - (min_train + test_size) + 1):
        for train_end_pos in range(start_pos + min_train, n - test_size + 1):
            train_start = idx[start_pos]
            train_end   = idx[train_end_pos - 1]
            test_start  = idx[train_end_pos]
            f.write(f"{train_start:%Y-%m-%d}\t{train_end:%Y-%m-%d}\t{test_size}\t{test_start:%Y-%m-%d}\n")

print("wrote splits.tsv:", sum(1 for _ in open("splits.tsv",encoding="utf-8")), "splits")
PY

while IFS=$'\t' read -r TRAIN_START TRAIN_END TEST_SIZE TEST_START; do
  python -tt pca_diag_unified.py one_split \
    --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
    --test-start "$TEST_START" --test-size "$TEST_SIZE" \
    --pc-use 3 --exog-mode all \
    --p-lag 1 \
    --no-irf --no-plots
done < splits.tsv

python - <<'PY'
import csv, subprocess

with open("splits.tsv", newline="") as f:
    r = csv.reader(f, delimiter="\t")
    for row in r:
        if not row or row[0].startswith("#"): 
            continue
        train_start, train_end, test_size, test_start = row
        cmd = [
            "python", "-tt", "pca_diag_unified.py", "one_split",
            "--train-start", train_start,
            "--train-end", train_end,
            "--test-start", test_start,
            "--test-size", test_size,
            "--pc-use", "3",
            "--exog-mode", "all",
            "--p-lag", "1",
            "--no-irf", "--no-plots",
        ]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=False)
PY

# 分割
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd

def load_dates(path: str) -> list[pd.Timestamp]:
    """
    優先順位:
      1) Date列があればそれ
      2) それ以外は「1列目(=index扱い)」を日付として読む
    """
    df = pd.read_csv(path)

    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
    else:
        # A列(1列目)が日付だと仮定
        d = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    d = d.dropna().drop_duplicates().sort_values()
    return list(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="A列(or Date列)に日付があるCSV")
    ap.add_argument("--out", default="splits.tsv")
    ap.add_argument("--per-year", type=int, default=4, help="1年あたりの点数（四半期=4）")
    ap.add_argument("--test-years", type=int, default=1, help="test年数（1年固定なら1）")
    ap.add_argument("--min-train-years", type=int, default=1, help="train最小年数")
    ap.add_argument("--max-train-years", type=int, default=10_000, help="train最大年数（制限したい時用）")
    args = ap.parse_args()

    dates = load_dates(args.data)
    N = len(dates)

    per_year = int(args.per_year)
    test_size = int(args.test_years) * per_year
    min_train = int(args.min_train_years) * per_year
    max_train = int(args.max_train_years) * per_year

    if N < min_train + test_size:
        raise SystemExit(f"データが短すぎる: N={N}, 最低必要={min_train+test_size}")

    rows = []
    for s in range(N):
        # train_len は per_year 刻み
        max_len_here = min(max_train, N - s - test_size)
        # 取りうる train_len: min_train, min_train+per_year, ...
        train_len = min_train
        while train_len <= max_len_here:
            train_start = dates[s]
            train_end   = dates[s + train_len - 1]
            test_start  = dates[s + train_len]

            rows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                str(test_size),
                test_start.strftime("%Y-%m-%d"),
            ))
            train_len += per_year

    # TSV出力（ヘッダ無し：while readでそのまま食える）
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"[OK] wrote {len(rows)} splits -> {args.out} (N_dates={N}, test_size={test_size}, per_year={per_year})")

if __name__ == "__main__":
    main()

# 実行
python make_splits.py --data ./data/all_q_merged.csv --out splits.tsv --per-year 4 --test-years 1 --min-train-years 1
