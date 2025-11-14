
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------
# Args
# --------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="reactionsCount",
                    help="reactionsCount | commentCount | shareCount")
    return ap.parse_args()


# --------------------
# Paths and IO utils
# --------------------
def _out_dir_for_target(target: str) -> Path:
    # ../../output/regression/<target>
    return (Path(__file__).resolve().parents[2] / "output" / "regression" / target)


def _evaluate(y_true, y_pred, model_name, split):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"  {split:>4} | {model_name:<25} MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    return {"split": split, "model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# --------------------
# Feature builder
# --------------------
def _build_rawonly_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build the STRICT baseline features:
      1) Post month extracted from a timestamp column, if available.
      2) Group indicator, one-hot encoded, if a group column exists.
    Absolutely NO cluster indicators here.
    """
    X = pd.DataFrame(index=df.index)

    # 1) Timestamp -> month
    ts_col = None
    for c in ["timestamp", "created_time", "post_created_time", "created_at"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        # month in 1..12, missing -> 0
        X["post_month"] = ts.dt.month.fillna(0).astype(int)
    else:
        # If no timestamp exists, keep a neutral constant so the model still runs
        X["post_month"] = 0

    # 2) Group indicator one-hot, prefer stable text id if available
    group_col = None
    for c in ["group", "group_name", "group_title", "group_id", "group_numeric_id"]:
        if c in df.columns:
            group_col = c
            break
    if group_col is not None:
        d = pd.get_dummies(df[group_col].astype(str), prefix="group", dtype=float)
        # Avoid exploding to thousands of columns silently. Keep all, this is baseline and typically only 2 groups.
        X = pd.concat([X, d], axis=1)

    # Safety checks, avoid any leakage
    for forbidden in ("reactionsCount", "commentCount", "shareCount", "cluster_id", "cluster"):
        assert forbidden not in X.columns, f"Label or cluster leakage detected: {forbidden}"

    # Fill remaining NaNs
    X = X.fillna(0.0)
    return X


# --------------------
# Main
# --------------------
def main():
    args = _parse_args()
    target = args.target
    out_dir = _out_dir_for_target(target)

    # Load frozen splits produced earlier
    train = pd.read_csv(out_dir / "train.csv")
    dev   = pd.read_csv(out_dir / "dev.csv")
    test  = pd.read_csv(out_dir / "test.csv")

    # Targets
    y_tr = train[target].astype(float).to_numpy()
    y_dv = dev[target].astype(float).to_numpy()
    y_te = test[target].astype(float).to_numpy()

    # Features for the STRICT baseline
    X_tr = _build_rawonly_features(train, target)
    X_dv = _build_rawonly_features(dev,   target)
    X_te = _build_rawonly_features(test,  target)

    # Fit OLS on TRAIN only
    model = LinearRegression()
    model.fit(X_tr.to_numpy(), y_tr)

    # Evaluate on dev, test, train for reference
    rows = []
    for split_name, X, y in [("dev", X_dv, y_dv), ("test", X_te, y_te), ("train", X_tr, y_tr)]:
        y_pred = model.predict(X.to_numpy())
        rows.append(_evaluate(y, y_pred, "BaselineFinal_RAWONLY", split_name))

    # Save metrics and predictions
    (out_dir / "metrics_baseline_final_rawonly.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig"),
        encoding="utf-8-sig"
    )
    pd.DataFrame({
        "post_id": test.get("post_id", pd.Series(np.arange(len(y_te)))),
        "y_true": y_te,
        "y_pred_baseline_rawonly": model.predict(X_te.to_numpy())
    }).to_csv(out_dir / "test_predictions_baseline_rawonly.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved RAWONLY baseline artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
