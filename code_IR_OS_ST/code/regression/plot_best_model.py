# plot_best_model.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import parse_target, out_dir_for_target

# known prediction files and their expected columns, in priority order
PRED_FILES = [
    ("test_predictions_tree_with_cluster_log.csv", ["y_pred_tree_with_cluster_log"]),
    ("test_predictions_tree_no_cluster_log.csv", ["y_pred_tree_log"]),
    ("test_predictions_tree_with_cluster.csv",    ["y_pred_tree_with_cluster"]),
    ("test_predictions_tree_no_cluster.csv",      ["y_pred_tree"]),
    ("test_predictions_baseline.csv",             ["y_pred_linear", "y_pred_mean"]),
]

def pick_pred_file(data_dir: Path, model_name: str):
    """Pick the right predictions file and column, first by model name, then by fallback list."""
    name = model_name or ""
    prefer = []
    if "withCluster_LOG" in name:
        prefer = [PRED_FILES[0]]
    elif "noCluster_LOG" in name:
        prefer = [PRED_FILES[1]]
    elif "withCluster" in name:
        prefer = [PRED_FILES[2]]
    elif "noCluster" in name:
        prefer = [PRED_FILES[3]]
    else:
        prefer = [PRED_FILES[4]]

    tried = []
    for filename, cols in prefer + PRED_FILES:
        f = data_dir / filename
        tried.append(str(f))
        if f.exists():
            df_head = pd.read_csv(f, nrows=1)
            for c in cols:
                if c in df_head.columns:
                    return f, c
    raise FileNotFoundError("Could not find predictions file. Looked for:\n" + "\n".join(tried))

def main():
    # which target to plot, default reactionsCount, can be overridden by --target <name>
    target = parse_target("reactionsCount")
    out_dir = out_dir_for_target(target)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick best model by RMSE then MAE
    metrics_path = out_dir / "metrics_all.csv"
    if not metrics_path.exists():
        print(f"[{target}] metrics_all.csv not found at {metrics_path}. run collect_metrics.py first.", file=sys.stderr)
        sys.exit(1)

    m = pd.read_csv(metrics_path)
    if m.empty:
        print(f"[{target}] metrics_all.csv is empty.", file=sys.stderr)
        sys.exit(1)

    best = m.sort_values(["RMSE", "MAE"], ascending=[True, True]).iloc[0]
    best_name = str(best["model"])

    # load predictions for the chosen model
    pred_file, pred_col = pick_pred_file(out_dir, best_name)
    preds = pd.read_csv(pred_file)

    y_true_col = "y_true"
    if y_true_col not in preds.columns:
        cands = [c for c in preds.columns if c.lower().startswith("y_true")]
        if not cands:
            print(f"[{target}] y_true column not found in {pred_file}", file=sys.stderr)
            sys.exit(1)
        y_true_col = cands[0]

    y_true = preds[y_true_col].astype(float).to_numpy()
    y_pred = preds[pred_col].astype(float).to_numpy()

    # metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # pretty plot
    plt.figure(figsize=(8, 8))
    mn = 0.0
    mx = 1.05 * max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.scatter(y_true, y_pred, s=18, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2, color="gray", label="y = x")

    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"[{target}] Actual vs Predicted, best model: {best_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "actual_vs_pred_best.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[{target}] Saved plot to {out_path}")

if __name__ == "__main__":
    main()
