# ridge_tabular_sweep.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------
# Args
# -----------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="reactionsCount",
                    help="reactionsCount | commentCount | shareCount")
    ap.add_argument("--alphas", type=str, default="1,10,100",
                    help="comma-separated list of alphas, e.g. '0.1,1,10,100'")
    ap.add_argument("--use-cluster", type=int, default=1,
                    help="1 = add one-hot of cluster_id if exists, 0 = ignore cluster_id")
    args = ap.parse_args()
    # parse alphas into list of floats
    try:
        args.alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip() != ""]
    except Exception:
        args.alphas = [1.0, 10.0, 100.0]
    if not args.alphas:
        args.alphas = [1.0, 10.0, 100.0]
    return args


# -----------------------
# Utils
# -----------------------
def _out_dir_for_target(target: str) -> Path:
    # Match the existing project structure: ../../output/regression/<target>
    return (Path(__file__).resolve().parents[2] / "output" / "regression" / target)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"  {split:>8} | {model_name:<24} MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    return {"split": split, "model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}



def _build_X(df: pd.DataFrame, target: str, use_cluster: bool) -> pd.DataFrame:


    X = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in ["post_id", "Unnamed: 0", target, "reactionsCount", "commentCount", "shareCount", "cluster_id"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")


    if use_cluster and (("cluster_id" in df.columns) or ("cluster" in df.columns)):
        w = df.copy()
        if "cluster_id" not in w.columns and "cluster" in w.columns:
            w = w.rename(columns={"cluster": "cluster_id"})
        if "cluster_id" in w.columns:
            d = pd.get_dummies(w["cluster_id"], prefix="cluster_id", dtype=float, dummy_na=False)
            if d.shape[1] > 0:
                X = pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)


    if X.shape[1] == 0:
        X["bias_only"] = 1.0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))


    for forbidden in {"reactionsCount", "commentCount", "shareCount", "cluster_id"}:
        assert forbidden not in X.columns, f"Leakage or raw cluster_id found in features: {forbidden}"

    return X

# -----------------------
# Main
# -----------------------
def main():
    args = _parse_args()
    target = args.target
    out_dir = _out_dir_for_target(target)

    # Load frozen splits
    train = pd.read_csv(out_dir / "train.csv")
    dev   = pd.read_csv(out_dir / "dev.csv")
    test  = pd.read_csv(out_dir / "test.csv")

    y_tr = train[target].astype(float).to_numpy()
    y_dv = dev[target].astype(float).to_numpy()
    y_te = test[target].astype(float).to_numpy()

    X_tr = _build_X(train, target, use_cluster=bool(args.use_cluster))
    X_dv = _build_X(dev,   target, use_cluster=bool(args.use_cluster))
    X_te = _build_X(test,  target, use_cluster=bool(args.use_cluster))

    print(f"\n=== {target} | Ridge (Tabular{' + Cluster' if args.use_cluster else ''}) alpha sweep ===")
    # Sweep alphas on dev RMSE (fit on train)
    best_alpha, best_rmse, best_model = None, float("inf"), None
    dev_rows = []
    for a in args.alphas:
        mdl = Ridge(alpha=float(a), random_state=42)
        mdl.fit(X_tr, y_tr)
        pred_dv = mdl.predict(X_dv)
        rmse_dv = float(np.sqrt(mean_squared_error(y_dv, pred_dv)))
        name = f"Ridge_TabularOnly_a{a}"
        dev_rows.append(_evaluate(y_dv, pred_dv, name, "dev"))
        if rmse_dv < best_rmse:
            best_alpha, best_rmse, best_model = a, rmse_dv, mdl

    assert best_model is not None, "No model trained in alpha sweep."
    print(f"\n[best] alpha={best_alpha} (dev RMSE={best_rmse:.3f})")

    # Refit best alpha on train+dev and evaluate on test
    trdev = pd.concat([train, dev], ignore_index=True)
    y_trdv = trdev[target].astype(float).to_numpy()
    X_trdv = _build_X(trdev, target, use_cluster=bool(args.use_cluster))

    final = Ridge(alpha=float(best_alpha), random_state=42)
    final.fit(X_trdv, y_trdv)

    # Eval on train+dev (for reference) and on test
    rows = []
    name = f"Ridge_TabularOnly_a{best_alpha}"
    rows.append(_evaluate(y_trdv, final.predict(X_trdv), name, "train+dev"))
    rows.append(_evaluate(y_te,    final.predict(X_te),   name, "test"))

    # Save: best metrics (single file), plus keep the per-alpha dev lines appended
    metrics_path = out_dir / "metrics_ridge_tabular_best.csv"
    pd.DataFrame(dev_rows + rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # Save test predictions
    pd.DataFrame({
        "post_id": test["post_id"] if "post_id" in test.columns else np.arange(len(test)),
        "y_true": y_te,
        f"y_pred_{name}": final.predict(X_te)
    }).to_csv(out_dir / f"test_predictions_{name}.csv", index=False, encoding="utf-8-sig")

    # Save coefficients (from the final train+dev fit)
    coefs = pd.DataFrame({
        "feature": X_trdv.columns,
        "coef": final.coef_.astype(float)
    })
    coefs["abs_coef"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False)
    coefs.to_csv(out_dir / "ridge_tabular_best_coefficients.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved coefficients to: {out_dir / 'ridge_tabular_best_coefficients.csv'}")
    print(f"Saved predictions to: {out_dir / f'test_predictions_{name}.csv'}")
    print(f"Done for target: {target}")


if __name__ == "__main__":
    main()
