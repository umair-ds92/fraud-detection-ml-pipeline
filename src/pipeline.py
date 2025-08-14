#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fraud Detection End-to-End Pipeline

Usage:
  python src/pipeline.py --download --model xgb --balance smote --threshold 0.5

Data:
  Kaggle: mlg-ulb/creditcardfraud
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from joblib import dump

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"

for d in [DATA_DIR, OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def download_from_kaggle():
    # requires Kaggle CLI & env vars KAGGLE_USERNAME, KAGGLE_KEY
    # Dataset: mlg-ulb/creditcardfraud
    print("[INFO] Downloading dataset via Kaggle CLI...")
    cmd = f'kaggle datasets download -d mlg-ulb/creditcardfraud -p "{DATA_DIR}" --force'
    code = os.system(cmd)
    if code != 0:
        print("[WARN] Kaggle download failed. Make sure Kaggle CLI is installed and credentials are set.")
        return
    # unzip
    import zipfile
    zips = list(DATA_DIR.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        z.unlink(missing_ok=True)

def load_data():
    # Expect 'creditcard.csv' in DATA_DIR
    csv_path = DATA_DIR / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run with --download or place the file manually.")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def basic_eda(df):
    ax = df['Class'].value_counts().plot(kind='bar')
    ax.set_title("Class balance (0=legit, 1=fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png")
    plt.close()

    plt.hist(np.log1p(df['Amount']), bins=50)
    plt.title("Log(Amount+1) distribution")
    plt.xlabel("log(Amount+1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "amount_log_hist.png")
    plt.close()

def prepare_xy(df):
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)
    scaler = StandardScaler()
    X[['Amount','Time']] = scaler.fit_transform(X[['Amount','Time']])
    return X, y

def get_model(name: str):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    if name == "xgb":
        if not XGB_OK:
            print("[WARN] xgboost not installed; falling back to RandomForest.")
            return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist"
        )
    raise ValueError("Unknown model; choose from: logreg, rf, xgb")

def build_pipeline(model_name: str, balance: str):
    steps = []
    if balance == "smote":
        steps.append(("smote", SMOTE(random_state=42)))
    elif balance == "undersample":
        steps.append(("under", RandomUnderSampler(random_state=42)))
    steps.append(("model", get_model(model_name)))
    return ImbPipeline(steps=steps)

def evaluate_and_save(model, X_test, y_test, y_scores, threshold=0.5, prefix="model"):
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    with open(TAB_DIR / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.plot([0,1],[0,1],'--')
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_test, y_scores)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_roc.png")
    plt.close()

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_pr.png")
    plt.close()

    sweep = pd.DataFrame({
        "threshold": np.r_[thresholds, 1.0],
        "precision": precision,
        "recall": recall
    })
    sweep.to_csv(TAB_DIR / "threshold_sweep.csv", index=False)

    kpi = pd.DataFrame([{
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_threshold": float(precision[(np.abs(np.r_[thresholds,1.0] - threshold)).argmin()]),
        "recall_at_threshold": float(recall[(np.abs(np.r_[thresholds,1.0] - threshold)).argmin()]),
        "threshold": threshold
    }])
    kpi.to_csv(TAB_DIR / "dashboard_metrics.csv", index=False)

def run_supervised(args):
    df = load_data()
    basic_eda(df)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline(args.model, args.balance)
    pipe.fit(X_train, y_train)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_scores = pipe.predict_proba(X_test)[:,1]
    else:
        y_scores = pipe.decision_function(X_test)

    evaluate_and_save(pipe, X_test, y_test, y_scores, threshold=args.threshold, prefix=f"{args.model}_{args.balance}")
    dump(pipe, MODEL_DIR / f"{args.model}_{args.balance}.joblib")

    scored = X_test.copy()
    scored["label"] = y_test.values
    scored["score"] = y_scores
    scored = scored.sample(n=min(10000, len(scored)), random_state=42)
    scored.to_csv(TAB_DIR / "scored_transactions_sample.csv", index=False)

def run_isolation_forest(args):
    df = load_data()
    basic_eda(df)
    X, y = prepare_xy(df)

    legit = X[y==0]
    iso = IsolationForest(
        n_estimators=300, contamination="auto", random_state=42, n_jobs=-1
    )
    iso.fit(legit)

    scores = -iso.decision_function(X)  # higher => more anomalous
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    evaluate_and_save(iso, X, y, scores, threshold=args.threshold, prefix="isoforest")

    out = X.copy()
    out["label"] = y.values
    out["score"] = scores
    out = out.sample(n=min(10000, len(out)), random_state=42)
    out.to_csv(TAB_DIR / "scored_transactions_sample.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--model", default="rf", choices=["logreg","rf","xgb","isoforest"], help="Model choice")
    parser.add_argument("--balance", default="smote", choices=["none","smote","undersample"], help="Imbalance strategy")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--fn-cost", type=float, default=10.0, help="Cost weight for false negatives")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="Cost weight for false positives")
    args = parser.parse_args()

    if args.download:
        download_from_kaggle()

    if args.model == "isoforest":
        run_isolation_forest(args)
    else:
        run_supervised(args)

if __name__ == "__main__":
    main()
