"""
AutoThink V4 vs FLAML vs AutoGluon — Fair Benchmark
Same data, same folds, same metric, same time budget.
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

warnings.filterwarnings('ignore')

# ======================================================================
# BENCHMARK CONFIG
# ======================================================================

TIME_BUDGET = 60  # seconds per tool (fair)
SEED = 42

# ======================================================================
# DATASETS
# ======================================================================

def load_heart_disease():
    df = pd.read_csv('playground-series-s6e2/train.csv')
    df = df.sample(n=10000, random_state=SEED).reset_index(drop=True)
    target = 'Heart Disease'
    task = 'binary'
    metric_name = 'AUC'
    return df, target, task, metric_name, 'Heart Disease (binary, 10K)'

def load_loan():
    df = pd.read_csv('playground-series-s5e11/train.csv')
    df = df.sample(n=10000, random_state=SEED).reset_index(drop=True)
    target = 'loan_paid_back'
    task = 'binary'
    metric_name = 'AUC'
    return df, target, task, metric_name, 'Loan Repayment (binary, 10K)'

def load_regression():
    """Synthetic but realistic regression task."""
    np.random.seed(SEED)
    n = 5000
    df = pd.DataFrame({
        'sqft': np.random.uniform(500, 5000, n),
        'bedrooms': np.random.choice([1,2,3,4,5], n),
        'age': np.random.uniform(0, 50, n),
        'location_score': np.random.uniform(1, 10, n),
        'garage': np.random.choice([0, 1], n),
        'neighborhood': np.random.choice(['A','B','C','D'], n),
    })
    df['price'] = (
        df['sqft'] * 150
        + df['bedrooms'] * 20000
        + df['location_score'] * 30000
        - df['age'] * 2000
        + df['garage'] * 15000
        + np.random.randn(n) * 30000
    )
    return df, 'price', 'regression', 'RMSE', 'House Price (regression, 5K)'

DATASETS = [load_heart_disease, load_loan, load_regression]

# ======================================================================
# RUNNERS
# ======================================================================

def run_autothink(X_train, y_train, X_val, y_val, task, target_col, time_budget):
    """Run AutoThink V4."""
    from autothink.core.autothink_v4 import AutoThinkV4

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    t0 = time.time()
    model = AutoThinkV4(time_budget=time_budget, verbose=False)
    model.fit(train_df, target_col)
    train_time = time.time() - t0

    preds = model.predict(X_val)
    return preds, train_time


def run_flaml(X_train, y_train, X_val, y_val, task, target_col, time_budget):
    """Run FLAML."""
    from flaml import AutoML

    automl = AutoML()
    flaml_task = 'classification' if task != 'regression' else 'regression'
    flaml_metric = 'roc_auc' if task == 'binary' else ('log_loss' if task == 'multiclass' else 'rmse')

    t0 = time.time()
    automl.fit(
        X_train, y_train,
        task=flaml_task,
        metric=flaml_metric,
        time_budget=time_budget,
        verbose=0,
        seed=SEED,
    )
    train_time = time.time() - t0

    if task in ('binary', 'multiclass'):
        preds = automl.predict_proba(X_val)
        if task == 'binary':
            preds = preds[:, 1]
    else:
        preds = automl.predict(X_val)
    return preds, train_time


def run_autogluon(X_train, y_train, X_val, y_val, task, target_col, time_budget):
    """Run AutoGluon."""
    from autogluon.tabular import TabularPredictor

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    ag_task = 'binary' if task == 'binary' else ('multiclass' if task == 'multiclass' else 'regression')
    ag_metric = 'roc_auc' if task == 'binary' else ('log_loss' if task == 'multiclass' else 'root_mean_squared_error')

    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        problem_type=ag_task,
        eval_metric=ag_metric,
        verbosity=0,
    ).fit(
        train_df,
        time_limit=time_budget,
    )
    train_time = time.time() - t0

    if task in ('binary', 'multiclass'):
        preds = predictor.predict_proba(X_val)
        if task == 'binary':
            # AutoGluon returns a DataFrame with class columns
            pos_col = preds.columns[-1]
            preds = preds[pos_col].values
        else:
            preds = preds.values
    else:
        preds = predictor.predict(X_val).values
    return preds, train_time


# ======================================================================
# SCORING
# ======================================================================

def score(y_true, y_pred, task):
    if task == 'binary':
        return roc_auc_score(y_true, y_pred)
    elif task == 'multiclass':
        return -log_loss(y_true, y_pred)
    else:
        return np.sqrt(mean_squared_error(y_true, y_pred))


# ======================================================================
# MAIN BENCHMARK
# ======================================================================

def main():
    print("=" * 80)
    print("  BENCHMARK: AutoThink V4 vs FLAML vs AutoGluon")
    print(f"  Time budget: {TIME_BUDGET}s per tool | Seed: {SEED}")
    print("=" * 80)

    all_results = []

    for dataset_fn in DATASETS:
        df, target, task, metric_name, dataset_name = dataset_fn()
        print(f"\n{'─' * 80}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Shape: {df.shape}  |  Task: {task}  |  Metric: {metric_name}")
        print(f"{'─' * 80}")

        # Prepare X, y
        X = df.drop(columns=[target])
        y = df[target]

        # Encode string target for sklearn compatibility
        label_map = None
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
            label_map = le

        # Single 80/20 split (same for all tools)
        np.random.seed(SEED)
        if task != 'regression':
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y)
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=SEED)

        # Need original target for autothink (it does its own encoding)
        if label_map is not None:
            y_train_orig = pd.Series(label_map.inverse_transform(y_train), index=y_train.index)
            y_val_orig = pd.Series(label_map.inverse_transform(y_val), index=y_val.index)
        else:
            y_train_orig = y_train
            y_val_orig = y_val

        runners = [
            ('AutoThink V4', lambda: run_autothink(X_train, y_train_orig, X_val, y_val_orig, task, target, TIME_BUDGET)),
            ('FLAML',        lambda: run_flaml(X_train, y_train, X_val, y_val, task, target, TIME_BUDGET)),
            ('AutoGluon',    lambda: run_autogluon(X_train, y_train_orig, X_val, y_val_orig, task, target, TIME_BUDGET)),
        ]

        results = {}
        for name, runner in runners:
            print(f"\n  Running {name}...", end=' ', flush=True)
            try:
                preds, elapsed = runner()
                s = score(y_val, preds, task)
                results[name] = {'score': s, 'time': elapsed}
                print(f"{metric_name}={s:.5f}  ({elapsed:.1f}s)")
            except Exception as e:
                results[name] = {'score': None, 'time': None}
                print(f"FAILED: {e}")

        # Summary table for this dataset
        print(f"\n  {'Tool':<20} {metric_name:>12} {'Time':>10}")
        print(f"  {'─'*44}")

        scored = [(n, r) for n, r in results.items() if r['score'] is not None]
        if task == 'regression':
            scored.sort(key=lambda x: x[1]['score'])  # lower RMSE = better
        else:
            scored.sort(key=lambda x: -x[1]['score'])  # higher AUC = better

        for rank, (name, r) in enumerate(scored):
            marker = ' <-- BEST' if rank == 0 else ''
            score_str = f"{r['score']:.5f}"
            time_str = f"{r['time']:.1f}s"
            print(f"  {name:<20} {score_str:>12} {time_str:>10}{marker}")

        all_results.append((dataset_name, results))

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    print(f"\n{'=' * 80}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 80}")

    wins = {'AutoThink V4': 0, 'FLAML': 0, 'AutoGluon': 0}

    for dataset_name, results in all_results:
        scored = [(n, r) for n, r in results.items() if r['score'] is not None]
        if not scored:
            continue
        # determine best
        task_for_ds = 'regression' if 'regression' in dataset_name.lower() or 'price' in dataset_name.lower() or 'RMSE' in dataset_name else 'classification'
        if task_for_ds == 'regression':
            best = min(scored, key=lambda x: x[1]['score'])
        else:
            best = max(scored, key=lambda x: x[1]['score'])
        wins[best[0]] += 1

    print(f"\n  Wins across {len(all_results)} datasets:")
    for tool, w in sorted(wins.items(), key=lambda x: -x[1]):
        bar = '#' * (w * 10)
        print(f"    {tool:<20} {w} win(s)  {bar}")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
