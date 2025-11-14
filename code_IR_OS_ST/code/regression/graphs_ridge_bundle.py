# graphs_ridge_bundle.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

TARGETS = ["reactionsCount","commentCount","shareCount"]

def find_best_ridge(out_dir: Path):
    """Load best ridge row for test split and derive prediction file/column."""
    mpath = out_dir / "metrics_ridge_tabular_best.csv"
    if not mpath.exists():
        return None
    m = pd.read_csv(mpath)
    row = m.loc[m["split"].astype(str).str.lower()=="test"].iloc[0]
    model = str(row["model"])
    # prediction file & column follow our ridge saver
    pred_file = out_dir / f"test_predictions_{model}.csv"
    pred_col = f"y_pred_{model}"
    return {"model": model, "pred_file": pred_file, "pred_col": pred_col}

def load_baseline_metrics(out_dir: Path):
    for fname in ["metrics_baseline_final_rawonly.csv","metrics_baseline_final.csv"]:
        p = out_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            te = df.loc[df["split"].astype(str).str.lower()=="test"]
            if not te.empty:
                return te.iloc[0]
    return None

def nice_name(t):
    return {"reactionsCount":"Reactions","commentCount":"Comments","shareCount":"Shares"}.get(t,t)

def make_scatter_actual_vs_pred(out_dir: Path, target: str, y_true, y_pred, model_name: str):
    plt.figure(figsize=(7,7))
    mn = 0.0
    mx = 1.05 * max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)
    plt.plot([mn,mx],[mn,mx], linestyle="--", linewidth=2)
    plt.xlim(mn,mx); plt.ylim(mn,mx)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{nice_name(target)} – Actual vs Predicted\n{model_name}")
    plt.tight_layout()
    (out_dir / "plot_actual_vs_pred_ridge.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "plot_actual_vs_pred_ridge.png", dpi=180)
    plt.close()

def make_residual_hist(out_dir: Path, target: str, y_true, y_pred):
    res = y_true - y_pred
    plt.figure(figsize=(7,5))
    plt.hist(res, bins=40)
    plt.xlabel("Residual (y - yhat)"); plt.ylabel("Frequency")
    plt.title(f"{nice_name(target)} – Residuals (test)")
    plt.tight_layout()
    (out_dir / "plot_residuals_hist_ridge.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "plot_residuals_hist_ridge.png", dpi=180)
    plt.close()

def make_coef_bar(out_dir: Path, target: str):
    p = out_dir / "ridge_tabular_best_coefficients.csv"
    if not p.exists(): return
    coef = pd.read_csv(p)
    if "abs_coef" not in coef.columns:
        coef["abs_coef"] = coef["coef"].abs()
    top = coef.sort_values("abs_coef", ascending=False).head(15)
    plt.figure(figsize=(8,6))
    y = np.arange(len(top))[::-1]
    plt.barh(y, top["abs_coef"].values[::-1])
    plt.yticks(y, top["feature"].values[::-1])
    plt.xlabel("|coef|"); plt.title(f"{nice_name(target)} – Top 15 Ridge coefficients")
    plt.tight_layout()
    (out_dir / "plot_ridge_top15_coefs.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "plot_ridge_top15_coefs.png", dpi=180)
    plt.close()

def make_rmse_compare_bar(out_dir: Path, target: str, ridge_rmse: float):
    base_row = load_baseline_metrics(out_dir)
    if base_row is None: return
    try:
        base_rmse = float(base_row["RMSE"])
    except Exception:
        return
    plt.figure(figsize=(5,4))
    xs = ["Baseline","Ridge"]
    vals = [base_rmse, ridge_rmse]
    plt.bar(xs, vals)
    plt.ylabel("RMSE (test)")
    plt.title(f"{nice_name(target)} – RMSE (test)")
    for i,v in enumerate(vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    (out_dir / "plot_rmse_compare_test.png").unlink(missing_ok=True)
    plt.savefig(out_dir / "plot_rmse_compare_test.png", dpi=180)
    plt.close()

def main():
    root = Path("../../output/regression")
    for t in TARGETS:
        out = root / t
        out.mkdir(parents=True, exist_ok=True)
        info = find_best_ridge(out)
        if info is None or not info["pred_file"].exists():
            print(f"[{t}] Missing ridge outputs, skipping.")
            continue
        preds = pd.read_csv(info["pred_file"])
        y_col = "y_true" if "y_true" in preds.columns else [c for c in preds.columns if c.lower().startswith("y_true")][0]
        y_true = preds[y_col].astype(float).to_numpy()
        y_pred = preds[info["pred_col"]].astype(float).to_numpy()

        # 1) scatter
        make_scatter_actual_vs_pred(out, t, y_true, y_pred, info["model"])
        # 2) residuals
        make_residual_hist(out, t, y_true, y_pred)
        # 3) coef bar
        make_coef_bar(out, t)
        # 4) rmse compare
        # try to read ridge rmse from metrics file
        m = pd.read_csv(out / "metrics_ridge_tabular_best.csv")
        ridge_test_rmse = float(m.loc[m["split"].astype(str).str.lower()=="test","RMSE"].iloc[0])
        make_rmse_compare_bar(out, t, ridge_test_rmse)

        print(f"[{t}] Saved plots to {out.resolve()}")

if __name__ == "__main__":
    main()
