"""AutoThink quickstart — fit/predict on sklearn datasets."""

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load data into a DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# One-line AutoML
from autothink import fit

model = fit(df, target="target", time_budget=60)

# Predict on new data
test_df = df.drop(columns=["target"]).iloc[:5]
predictions = model.predict(test_df)

print(f"CV score: {model.cv_score:.4f}")
print(f"Predictions: {predictions}")
