from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Targets we treat as engagement labels (never allowed as features)
TARGETS = ["reactionsCount", "commentCount", "shareCount"]

def parse_target(default: str = "reactionsCount") -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=default, help="target column name")
    return p.parse_args().target

def out_dir_for_target(target: str) -> Path:
    d = Path("../../output/regression") / target
    d.mkdir(parents=True, exist_ok=True)
    return d

def numeric_feats(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Safe numeric features builder:
    - Drops all engagement columns (including the current target)
    - Drops IDs / unnamed columns
    - Drops raw cluster columns ('cluster_id' / 'cluster'); cluster one-hots, if desired,
      will be added explicitly elsewhere.
    """
    num = df.select_dtypes(include=[np.number]).copy()

    # Drop ALL engagement columns + ids + raw cluster columns
    drop = list(TARGETS) + ["post_id", "Unnamed: 0", "cluster_id", "cluster"]
    num = num.drop(columns=[c for c in drop if c in num.columns], errors="ignore")

    # Safety fill + fallback bias feature
    num = num.fillna(num.median(numeric_only=True))
    if num.shape[1] == 0:
        num["bias_only"] = 1.0

    return num
