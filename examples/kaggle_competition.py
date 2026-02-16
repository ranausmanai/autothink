"""AutoThink Kaggle competition pipeline.

Demonstrates how to use AutoThink for a Kaggle binary-classification
competition with CSV train/test files and a submission output.

Usage:
    python kaggle_competition.py --train train.csv --test test.csv \
        --target "Heart Disease" --id id --out submission.csv
"""

import argparse
import pandas as pd
from autothink import fit


def main():
    parser = argparse.ArgumentParser(description="AutoThink Kaggle runner")
    parser.add_argument("--train", required=True, help="Path to train CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id", default="id", help="ID column in test set")
    parser.add_argument("--out", default="submission.csv", help="Output path")
    parser.add_argument("--budget", type=int, default=600, help="Time budget (s)")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    print(f"Train: {train.shape}  |  Test: {test.shape}")

    model = fit(train, target=args.target, time_budget=args.budget, verbose=True)

    predictions = model.predict(test)

    submission = pd.DataFrame({args.id: test[args.id], args.target: predictions})
    submission.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
