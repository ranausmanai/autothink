"""Benchmark AutoThink V4 against FLAML and AutoGluon.

Requires optional dependencies: pip install flaml autogluon

Usage:
    python benchmark.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

SEED = 42
TIME_BUDGET = 60


def load_binary():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, "target", "binary", "AUC", "Breast Cancer (binary)"


def load_regression():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, "target", "regression", "RMSE", "Diabetes (regression)"


DATASETS = [load_binary, load_regression]


def score(y_true, y_pred, task):
    if task == "binary":
        return roc_auc_score(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run_autothink(X_train, y_train, X_val, task, target):
    from autothink import AutoThinkV4

    train_df = X_train.copy()
    train_df[target] = y_train.values
    t0 = time.time()
    model = AutoThinkV4(time_budget=TIME_BUDGET, verbose=False)
    model.fit(train_df, target)
    elapsed = time.time() - t0
    preds = model.predict(X_val)
    return preds, elapsed


def run_flaml(X_train, y_train, X_val, task, target):
    from flaml import AutoML

    automl = AutoML()
    flaml_task = "classification" if task != "regression" else "regression"
    t0 = time.time()
    automl.fit(X_train, y_train, task=flaml_task, time_budget=TIME_BUDGET, verbose=0)
    elapsed = time.time() - t0
    if task == "binary":
        preds = automl.predict_proba(X_val)[:, 1]
    else:
        preds = automl.predict(X_val)
    return preds, elapsed


def main():
    print(f"{'=' * 70}")
    print(f"  AutoThink V4 Benchmark  (budget={TIME_BUDGET}s, seed={SEED})")
    print(f"{'=' * 70}")

    for dataset_fn in DATASETS:
        df, target, task, metric_name, name = dataset_fn()
        X = df.drop(columns=[target])
        y = df[target]

        stratify = y if task != "regression" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=stratify
        )

        print(f"\n--- {name} ({task}) ---")

        runners = [
            ("AutoThink V4", run_autothink),
            ("FLAML", run_flaml),
        ]

        for runner_name, runner_fn in runners:
            try:
                preds, elapsed = runner_fn(X_train, y_train, X_val, task, target)
                s = score(y_val, preds, task)
                print(f"  {runner_name:<20} {metric_name}={s:.5f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  {runner_name:<20} FAILED: {e}")


if __name__ == "__main__":
    main()
