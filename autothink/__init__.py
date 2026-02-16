"""
AutoThink - Self-Building ML System

Throw any data, get a working model.
"""

import logging
import sys

from .core.autothink_v2 import AutoThinkV2, AutoThinkResult
from .core.autothink_v3 import AutoThinkV3, fit_v3
from .core.autothink_v4 import AutoThinkV4, fit_v4
from .core.preprocessing import IntelligentPreprocessor, FeatureEngineer
from .core.meta_learning import MetaLearningDB, DatasetFingerprint
from .core.production import ModelExporter, ModelCard, DriftDetector
from .core.advanced import (
    CausalAutoML, ExplanationEngine, SmartEnsemble,
    UncertaintyQuantifier, AutomatedFeatureSelection
)
from .core.validation import DataValidator, LeakageDetector

__version__ = "4.0.0"
__all__ = [
    'AutoThinkV2',
    'AutoThinkV3',
    'AutoThinkV4',
    'fit',
    'fit_v3',
    'fit_v4',
    'AutoThinkResult',
    'IntelligentPreprocessor',
    'FeatureEngineer',
    'MetaLearningDB',
    'ModelCard',
    'ExplanationEngine',
    'DataValidator',
    'LeakageDetector',
    'setup_logging',
]

# Default: library is silent (NullHandler)
logging.getLogger(__name__).addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, stream=None):
    """Enable console logging for AutoThink.

    Call this once at the start of your script to see progress output.
    When ``verbose=True`` (the default), fit() also sets up a handler
    automatically so you usually don't need this.

    Args:
        level: Logging level (default ``logging.INFO``).
        stream: Output stream (default ``sys.stderr``).
    """
    if stream is None:
        stream = sys.stderr
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def fit(df, target: str, **kwargs):
    """
    One-line AutoML. Throw data, get model.

    Auto-detects binary classification, multiclass classification,
    or regression.  Trains a LightGBM + XGBoost + CatBoost ensemble
    with adaptive hyperparameters and optimised blend weights.

    Args:
        df: DataFrame with features and target
        target: Name of target column
        **kwargs: Passed to AutoThinkV4 (time_budget, verbose)

    Returns:
        AutoThinkV4 instance (call .predict(test_df) for predictions)

    Example:
        >>> import pandas as pd
        >>> from autothink import fit
        >>> df = pd.read_csv('data.csv')
        >>> result = fit(df, target='price')
        >>> predictions = result.predict(test_df)
    """
    verbose = kwargs.get("verbose", True)
    if verbose:
        # Ensure there's at least one visible handler when verbose=True
        pkg_logger = logging.getLogger(__name__)
        if not any(
            isinstance(h, logging.StreamHandler)
            for h in pkg_logger.handlers
            if not isinstance(h, logging.NullHandler)
        ):
            setup_logging(stream=sys.stdout)
    return fit_v4(df, target, **kwargs)
