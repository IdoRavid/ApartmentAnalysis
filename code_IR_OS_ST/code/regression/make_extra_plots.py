# make_extra_plots.py  (Python 3.9 compatible; correct ROOT based on __file__)
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ROOT = ../../output/regression relative to this file
ROOT = Path(__file__).resolve().parents[2] / "output" / "regression"

def autodetect_tag(dirpath: Path) -> str:
    """
    Pick the model tag to load predictions for:
    1) Prefer the 'model' value from metrics_ridge_tabular_best.csv (test row).
    2) Fallback to the first test_predictions_*.csv in the directory (incl. baseline).
    3) Explicit baseline fallback if present.
    """
    metrics = dirpath / "metrics_ridge_tabular_best.csv"
    if metrics.exists():
        dfm = pd.read_csv(metrics)
        row = dfm[dfm["split"].str.lower().str.contains("test")].tail(1)
        if row.empty:
            row = dfm.tail(1)
        tag = str(row["model"].iloc[0])
        if (dirpath / f"test_predictions_{tag}.csv").exists():
            return tag

    for p in dirpath.glob("test_predictions_*.csv"):
        return p.stem.replace("test_predictions_", "")

    baseline = dirpath / "test_predictions_baseline_rawonly.csv"
    if baseline.exists():
        return "baseline_rawonly"

    raise FileNotFoundError(f"No prediction files found in {dirpath}")

def load_preds(dirpath: Path, tag: Optional[str] = None):
    """
    Load y_true and y_pred arrays from the detected predictions file.
    Returns: (y_true, y_pred, tag_used)
    """
    if tag is None:
        tag = autodetect_tag(dirpath)
    df = pd.read_csv(dirpath / f"test_predictions_{tag}.csv")
    # baseline column name differs
    if tag == "baseline_rawonly":
        y_pred_col = "y_pred_baseline_rawonly"
    else:
        y_pred_col = [c for c in df.columns if c.startswith("y_pred_")][-1]
    return df["y_true"].to_numpy(), df[y_pred_col].to_numpy(), tag

def residuals_vs_fitted(base: Path, tag: Optional[str] = None, title_prefix: str = ""):
    y_true, y_hat, tag = load_preds(base, tag)
    resid = y_true - y_hat
    plt.figure()
    plt.scatter(y_hat, resid, s=8)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted (ŷ)")
    plt.ylabel("Residuals (y − ŷ)")
    plt.title(f"{title_prefix} – Residuals vs. Fitted")
    plt.tight_layout()
    plt.savefig(base / "plot_residuals_vs_fitted_ridge.png", dpi=180)
    plt.close()

def binned_actual_vs_pred(base: Path, tag: Optional[str] = None, title_prefix: str = "", bins: int = 10):
    y_true, y_hat, tag = load_preds(base, tag)
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(y_hat, q)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(y_hat.min(), y_hat.max(), bins + 1)
    idx = np.digitize(y_hat, edges, right=True)
    rows = []
    for b in range(1, len(edges)):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "y_true_mean": float(np.mean(y_true[mask])),
            "y_pred_mean": float(np.mean(y_hat[mask])),
            "count": int(mask.sum())
        })
    out = pd.DataFrame(rows)
    plt.figure()
    plt.plot(out["y_pred_mean"], out["y_true_mean"], marker="o")
    lim_min = min(out["y_pred_mean"].min(), out["y_true_mean"].min())
    lim_max = max(out["y_pred_mean"].max(), out["y_true_mean"].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("Predicted mean (per bin)")
    plt.ylabel("Actual mean (per bin)")
    plt.title(f"{title_prefix} – Binned Actual vs. Predicted")
    plt.tight_layout()
    plt.savefig(base / "plot_binned_avp_ridge.png", dpi=180)
    plt.close()

def error_by_cluster(base: Path, tag: Optional[str] = None, title_prefix: str = "", cluster_col: str = "cluster_id"):
    test_df = pd.read_csv(base / "test.csv")
    if cluster_col not in test_df.columns:
        print(f"[warn] {cluster_col} not in test.csv; skipping cluster plot for {base.name}.")
        return
    y_true, y_hat, tag = load_preds(base, tag)
    df = test_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_hat
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    grp = df.groupby(cluster_col, dropna=False)["abs_err"].mean().reset_index()
    grp = grp.sort_values("abs_err")
    plt.figure(figsize=(8, max(3, 0.35 * len(grp))))
    plt.barh(grp[cluster_col].astype(str), grp["abs_err"])
    plt.xlabel("Test MAE")
    plt.ylabel("Cluster")
    plt.title(f"{title_prefix} – Test MAE by Cluster")
    plt.tight_layout()
    plt.savefig(base / "plot_mae_by_cluster_ridge.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    targets = {
        "reactionsCount": "Reactions",
        "commentCount":   "Comments",
        "shareCount":     "Shares",
    }
    for target, title in targets.items():
        base = ROOT / target
        
        print(f"[info] looking in: {base}")

        has_preds = any(base.glob("test_predictions_*.csv")) or (base / "test_predictions_baseline_rawonly.csv").exists()
        if not has_preds:
            print(f"[{target}] skipped – no prediction files in {base}")
            continue
        residuals_vs_fitted(base, title_prefix=title)
        binned_actual_vs_pred(base, title_prefix=title)
        error_by_cluster(base, title_prefix=title)
        print(f"[{target}] Saved extra plots in {base}")
