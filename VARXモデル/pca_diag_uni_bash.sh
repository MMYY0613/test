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
