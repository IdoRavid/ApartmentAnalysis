import pandas as pd
import numpy as np
from pathlib import Path
from common import parse_target, out_dir_for_target

TARGET = parse_target("reactionsCount")
OUT = out_dir_for_target(TARGET)

# 1. splits summary
train = pd.read_csv(OUT/"train.csv"); dev = pd.read_csv(OUT/"dev.csv"); test = pd.read_csv(OUT/"test.csv")
pd.DataFrame([
    {"split":"train","rows":len(train)},
    {"split":"dev","rows":len(dev)},
    {"split":"test","rows":len(test)},
]).to_csv(OUT/"splits_summary.csv", index=False, encoding="utf-8-sig")

# 2. top10 feature importances of the winning log model with clusters
fi_path = OUT/"feature_importance_with_cluster_log.csv"
if fi_path.exists():
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False)
    fi.head(10).to_csv(OUT/"feature_importance_top10.csv", index=False, encoding="utf-8-sig")

# 3. per cluster summary on the full dataset
ds = pd.read_csv(Path("../../output/regression/dataset.csv"))
if "cluster_id" in ds.columns and TARGET in ds.columns:
    per_cl = ds.groupby("cluster_id")[TARGET].agg(
        count="count", mean="mean", median="median", std="std"
    ).reset_index().sort_values("count", ascending=False)
    per_cl.to_csv(OUT/"per_cluster_summary.csv", index=False, encoding="utf-8-sig")

# 4. hardest examples on test
pred_file = OUT/"test_predictions_tree_with_cluster_log.csv"
if pred_file.exists():
    preds = pd.read_csv(pred_file)
    pred_col = preds.filter(like="y_pred").columns[0]
    preds["abs_err"] = (preds["y_true"] - preds[pred_col]).abs()
    preds.sort_values("abs_err", ascending=False).head(30).to_csv(
        OUT/"sample_predictions.csv", index=False, encoding="utf-8-sig"
    )

print(f"[{TARGET}] Report tables saved in {OUT}")
