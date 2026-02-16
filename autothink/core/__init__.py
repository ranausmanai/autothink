"""AutoThink Core Module"""

from .autothink_v2 import AutoThinkV2, AutoThinkResult
from .autothink_v4 import AutoThinkV4, fit_v4
from .preprocessing import IntelligentPreprocessor, FeatureEngineer
from .meta_learning import MetaLearningDB, DatasetFingerprint, ExperimentRecord
from .production import ModelExporter, ModelCard, DriftDetector, APIGenerator
from .advanced import (
    CausalAutoML, ExplanationEngine, SmartEnsemble,
    UncertaintyQuantifier, AutomatedFeatureSelection
)

__all__ = [
    'AutoThinkV2',
    'AutoThinkV4',
    'fit_v4',
    'AutoThinkResult',
    'IntelligentPreprocessor',
    'FeatureEngineer',
    'MetaLearningDB',
    'DatasetFingerprint',
    'ExperimentRecord',
    'ModelExporter',
    'ModelCard',
    'DriftDetector',
    'APIGenerator',
    'CausalAutoML',
    'ExplanationEngine',
    'SmartEnsemble',
    'UncertaintyQuantifier',
    'AutomatedFeatureSelection',
]
