"""Tests for IntelligentPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from autothink.core.preprocessing import IntelligentPreprocessor


class TestHandlesMissingValues:
    def test_numeric_missing(self):
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10, 20, 30, 40]})
        y = pd.Series([0, 1, 0, 1])
        pp = IntelligentPreprocessor()
        result = pp.fit_transform(X, y)
        assert not result.isnull().any().any()

    def test_categorical_missing(self):
        X = pd.DataFrame({"cat": ["a", "b", None, "a"], "num": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])
        pp = IntelligentPreprocessor()
        result = pp.fit_transform(X, y)
        assert not result.isnull().any().any()


class TestHandlesMixedTypes:
    def test_numeric_and_categorical(self):
        X = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [10, 20, 30, 40, 50],
            "cat": ["a", "b", "c", "a", "b"],
        })
        y = pd.Series([0, 1, 0, 1, 0])
        pp = IntelligentPreprocessor()
        result = pp.fit_transform(X, y)
        assert result.shape[0] == 5
        assert result.select_dtypes(include=["object"]).shape[1] == 0


class TestHandlesCategoricalColumns:
    def test_low_cardinality_onehot(self):
        X = pd.DataFrame({"cat": ["a", "b", "c"] * 10, "num": range(30)})
        y = pd.Series([0, 1, 0] * 10)
        pp = IntelligentPreprocessor()
        result = pp.fit_transform(X, y)
        # one-hot should expand columns
        assert result.shape[1] > 2

    def test_transform_matches_fit(self):
        X = pd.DataFrame({"cat": ["a", "b", "c"] * 10, "num": range(30)})
        y = pd.Series([0, 1, 0] * 10)
        pp = IntelligentPreprocessor()
        pp.fit(X, y)
        result_fit = pp.transform(X)
        result_ft = pp.fit_transform(X, y)
        assert result_fit.shape == result_ft.shape
