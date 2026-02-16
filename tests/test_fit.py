"""Core integration tests for AutoThink fit/predict."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes


def _make_df(bunch, target_name="target"):
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[target_name] = bunch.target
    return df


# ------------------------------------------------------------------
# Binary classification
# ------------------------------------------------------------------

class TestBinaryClassification:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_df(load_breast_cancer())

    def test_fit_returns_model(self):
        from autothink import fit
        model = fit(self.df, target="target", time_budget=60, verbose=False)
        assert model is not None

    def test_predict_shape(self):
        from autothink import fit
        model = fit(self.df, target="target", time_budget=60, verbose=False)
        test_df = self.df.drop(columns=["target"]).iloc[:10]
        preds = model.predict(test_df)
        assert preds.shape[0] == 10

    def test_cv_score_reasonable(self):
        from autothink import fit
        model = fit(self.df, target="target", time_budget=60, verbose=False)
        assert model.cv_score > 0.8


# ------------------------------------------------------------------
# Multiclass classification
# ------------------------------------------------------------------

class TestMulticlassClassification:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_df(load_iris())

    def test_fit_detects_multiclass(self):
        from autothink import AutoThinkV4
        model = AutoThinkV4(time_budget=60, verbose=False)
        model.fit(self.df, target_col="target")
        assert model.task_info["task_type"] == "multiclass"

    def test_predict_shape(self):
        from autothink import AutoThinkV4
        model = AutoThinkV4(time_budget=60, verbose=False)
        model.fit(self.df, target_col="target")
        test_df = self.df.drop(columns=["target"]).iloc[:5]
        preds = model.predict(test_df)
        # multiclass returns (n_samples, n_classes) or (n_samples,)
        assert preds.shape[0] == 5


# ------------------------------------------------------------------
# Regression
# ------------------------------------------------------------------

class TestRegression:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = _make_df(load_diabetes())

    def test_fit_detects_regression(self):
        from autothink import AutoThinkV4
        model = AutoThinkV4(time_budget=60, verbose=False)
        model.fit(self.df, target_col="target")
        assert model.task_info["task_type"] == "regression"

    def test_predict_shape(self):
        from autothink import AutoThinkV4
        model = AutoThinkV4(time_budget=60, verbose=False)
        model.fit(self.df, target_col="target")
        test_df = self.df.drop(columns=["target"]).iloc[:5]
        preds = model.predict(test_df)
        assert preds.shape == (5,)


# ------------------------------------------------------------------
# Verbose flag
# ------------------------------------------------------------------

def test_verbose_false_does_not_raise():
    from autothink import fit
    df = _make_df(load_breast_cancer())
    model = fit(df, target="target", time_budget=60, verbose=False)
    assert model is not None
