import pandas as pd
from pathlib import Path
from common import parse_target, out_dir_for_target

TARGET = parse_target("reactionsCount")
OUT_DIR = out_dir_for_target(TARGET)

dfs = []
for p in sorted(OUT_DIR.glob("metrics_*.csv")):
    if p.name == "metrics_all.csv":
        continue
    df = pd.read_csv(p)
    df["source_file"] = p.name
    dfs.append(df)

allm = pd.concat(dfs, ignore_index=True)
allm = allm.sort_values(["RMSE","MAE"], ascending=[True, True]).reset_index(drop=True)
allm.to_csv(OUT_DIR / "metrics_all.csv", index=False, encoding="utf-8-sig")
print(f"[{TARGET}] Saved metrics_all.csv with {len(allm)} rows to {OUT_DIR}")
print(allm.head(10).to_string(index=False))
