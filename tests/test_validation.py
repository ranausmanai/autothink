"""Tests for DataValidator and LeakageDetector."""

import numpy as np
import pandas as pd
import pytest

from autothink.core.validation import DataValidator, LeakageDetector


class TestLeakageDetector:
    def test_catches_id_column(self):
        X = pd.DataFrame({
            "user_id": range(100),
            "feature": np.random.randn(100),
        })
        leaks = LeakageDetector.check_id_leakage(X)
        id_cols = [l["column"] for l in leaks]
        assert "user_id" in id_cols

    def test_no_false_positive_on_numeric_feature(self):
        X = pd.DataFrame({
            "value": np.random.randn(100),
            "other": np.random.randn(100),
        })
        leaks = LeakageDetector.check_id_leakage(X)
        assert len(leaks) == 0

    def test_target_leakage_detected(self):
        n = 200
        y = pd.Series(np.random.randint(0, 2, n))
        X = pd.DataFrame({
            "leak": y + np.random.randn(n) * 0.001,
            "clean": np.random.randn(n),
        })
        leaks = LeakageDetector.check_target_leakage(X, y, threshold=0.95)
        leak_cols = [l["column"] for l in leaks]
        assert "leak" in leak_cols


class TestDataValidator:
    def test_empty_dataset(self):
        X = pd.DataFrame()
        report = DataValidator.validate_dataset(X)
        assert not report["is_valid"]

    def test_valid_dataset(self):
        X = pd.DataFrame({
            "a": np.random.randn(200),
            "b": np.random.randn(200),
        })
        y = pd.Series(np.random.randint(0, 2, 200))
        report = DataValidator.validate_dataset(X, y)
        assert report["is_valid"]

    def test_flags_constant_features(self):
        X = pd.DataFrame({
            "constant": [1] * 50,
            "good": np.random.randn(50),
        })
        report = DataValidator.validate_dataset(X)
        has_constant_warning = any("onstant" in w for w in report["warnings"])
        assert has_constant_warning
