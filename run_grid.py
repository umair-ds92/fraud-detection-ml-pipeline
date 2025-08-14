# run_grid.py
import sys
import subprocess
from pathlib import Path
import pandas as pd
import math

ROOT = Path(__file__).resolve().parent
TAB_DIR = ROOT / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

models = ["logreg", "rf", "xgb", "isoforest"]
balances = ["none", "smote", "undersample"]
thresholds = [0.4, 0.7]

rows = []
first_run = False  # include --download only on the first run (comment out if you don't want auto-download)

def run_once(model, balance, threshold, first):
    """Run the pipeline once with given params; returns the parsed KPI row as dict."""
    cmd = [sys.executable, "src/pipeline.py"]
    if first:
        cmd.append("--download")
    cmd += ["--model", model, "--threshold", str(threshold)]

    # Isolation Forest is unsupervised; ignore balance options
    if model != "isoforest":
        cmd += ["--balance", balance]
        tag = f"{model}_{balance}_{threshold}"
    else:
        tag = f"{model}_none_{threshold}"

    print(f"\n[RUN] {tag}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Pipeline failed for {tag}")

    # Read the KPI row written by the pipeline
    kpi_path = TAB_DIR / "dashboard_metrics.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"Expected KPI file not found: {kpi_path}")

    kpi = pd.read_csv(kpi_path)
    if kpi.empty:
        raise ValueError("KPI CSV is empty.")

    # Take first row and augment with run context
    k = kpi.iloc[0].to_dict()
    k["model"] = model
    k["balance"] = balance if model != "isoforest" else "none"
    k["threshold"] = float(threshold)

    # Add F1 at chosen threshold (harmonic mean of precision & recall)
    p = k.get("precision_at_threshold", float("nan"))
    r = k.get("recall_at_threshold", float("nan"))
    if (p is not None and r is not None and not math.isnan(p) and not math.isnan(r) and (p + r) > 0):
        k["f1_at_threshold"] = 2 * p * r / (p + r)
    else:
        k["f1_at_threshold"] = float("nan")

    return k

# Run the grid
for m in models:
    if m == "isoforest":
        # Only one balance setting is meaningful
        b_list = ["none"]
    else:
        b_list = balances

    for t in thresholds:
        for b in b_list:
            k = run_once(m, b, t, first_run)
            first_run = False
            rows.append(k)

# Aggregate results
df = pd.DataFrame(rows)

# Order columns nicely
cols = [
    "model", "balance", "threshold",
    "roc_auc", "pr_auc",
    "precision_at_threshold", "recall_at_threshold", "f1_at_threshold"
]
df = df[cols]

# Round for readability
df_rounded = df.copy()
for c in ["roc_auc","pr_auc","precision_at_threshold","recall_at_threshold","f1_at_threshold"]:
    df_rounded[c] = df_rounded[c].astype(float).round(4)

# Save to CSV and Excel
out_csv = TAB_DIR / "kpi_grid.csv"
out_xlsx = TAB_DIR / "kpi_grid.xlsx"
df_rounded.to_csv(out_csv, index=False)
df_rounded.to_excel(out_xlsx, index=False)

# Show a readable table sorted by a business-friendly criterion:
# sort first by threshold (0.7 before 0.4), then by precision desc, then recall desc
sort_map = {0.7: 0, 0.4: 1}  # prefer 0.7 first
df_rounded["_thr_rank"] = df_rounded["threshold"].map(sort_map).fillna(99)
df_show = df_rounded.sort_values(by=["_thr_rank","precision_at_threshold","recall_at_threshold"], ascending=[True, False, False]).drop(columns=["_thr_rank"])

print("\n=== KPI GRID (Sorted: prefer threshold 0.7, then Precision, then Recall) ===")
print(df_show.to_string(index=False))

print(f"\nSaved KPI grid to:\n- {out_csv}\n- {out_xlsx}")
