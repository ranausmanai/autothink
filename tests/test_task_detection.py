"""Tests for TaskDetector."""

import numpy as np
import pandas as pd
import pytest

from autothink.core.autothink_v4 import TaskDetector


class TestTaskDetector:
    def test_binary_numeric(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        result = TaskDetector.detect(y)
        assert result["task_type"] == "binary"
        assert result["metric"] == "auc"
        assert result["n_classes"] == 2

    def test_binary_string(self):
        y = pd.Series(["Yes", "No", "Yes", "No", "Yes"])
        result = TaskDetector.detect(y)
        assert result["task_type"] == "binary"
        assert result["label_encoder"] is not None

    def test_multiclass(self):
        y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
        result = TaskDetector.detect(y)
        assert result["task_type"] == "multiclass"
        assert result["metric"] == "log_loss"
        assert result["n_classes"] == 3

    def test_multiclass_string(self):
        y = pd.Series(["cat", "dog", "bird"] * 5)
        result = TaskDetector.detect(y)
        assert result["task_type"] == "multiclass"
        assert result["label_encoder"] is not None

    def test_regression(self):
        y = pd.Series(np.random.randn(100))
        result = TaskDetector.detect(y)
        assert result["task_type"] == "regression"
        assert result["metric"] == "rmse"
        assert result["n_classes"] == 0

    def test_encoded_y_shape(self):
        y = pd.Series(["Presence", "Absence"] * 50)
        result = TaskDetector.detect(y)
        assert len(result["encoded_y"]) == 100
        assert set(result["encoded_y"].unique()) <= {0, 1}
