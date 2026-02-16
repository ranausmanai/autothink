"""
AutoThink v2 - Complete Implementation
Integrates all phases: Smart preprocessing, meta-learning, adaptive pipeline
"""

import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import time
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from .preprocessing import IntelligentPreprocessor, FeatureEngineer
from .meta_learning import MetaLearningDB, DatasetFingerprint, ExperimentRecord, AdaptivePipeline, HyperparameterTransfer

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AutoThinkResult:
    """Complete result from AutoThink."""
    model: Any
    model_name: str
    cv_score: float
    cv_std: float
    test_score: Optional[float]
    train_time: float
    preprocessor: IntelligentPreprocessor
    feature_engineer: Optional[FeatureEngineer]
    fingerprint: DatasetFingerprint
    all_results: List[Dict]
    feature_importance: Dict[str, float]


class AutoThinkV2:
    """
    Self-building ML system with intelligence.
    
    Features:
    - Smart preprocessing (Phase 2)
    - Meta-learning (Phase 3)
    - Adaptive model selection
    - Automatic feature engineering
    """
    
    def __init__(self, 
                 time_budget: int = 600,
                 target_score: float = 0.95,
                 use_meta_learning: bool = True,
                 use_feature_engineering: bool = True,
                 verbose: bool = True):
        """
        Initialize AutoThink.
        
        Args:
            time_budget: Maximum time in seconds
            target_score: Target CV score to achieve
            use_meta_learning: Use past experience
            use_feature_engineering: Create new features
            verbose: Print progress
        """
        self.time_budget = time_budget
        self.target_score = target_score
        self.use_meta_learning = use_meta_learning
        self.use_feature_engineering = use_feature_engineering
        self.verbose = verbose
        
        # Components
        self.preprocessor = None
        self.feature_engineer = None
        self.metadb = MetaLearningDB() if use_meta_learning else None
        self.adaptive = AdaptivePipeline(self.metadb, target_score) if use_meta_learning else None
        self.hp_transfer = HyperparameterTransfer(self.metadb) if use_meta_learning else None
        
        # Results
        self.fingerprint = None
        self.best_result = None
        self.all_results = []
        
    def fit(self, df: pd.DataFrame, target_col: str, 
            test_df: Optional[pd.DataFrame] = None) -> AutoThinkResult:
        """
        Main entry point: Build ML model automatically.
        
        Args:
            df: Training dataframe
            target_col: Target column name
            test_df: Optional test dataframe for final evaluation
        
        Returns:
            AutoThinkResult with model and metadata
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("=" * 70)
            logger.info("AUTOTHINK v2 - Self-Building ML System")
            logger.info("=" * 70)
        
        # Phase 1: Data Intelligence
        X, y, problem_type = self._prepare_data(df, target_col)
        
        # Phase 2: Smart Preprocessing
        X_processed = self._preprocess(X, y)
        
        # Phase 2b: Feature Engineering
        if self.use_feature_engineering:
            X_processed = self._engineer_features(X_processed, y)
        
        # Phase 3 & 4: Intelligent Model Selection & Training
        self._train_models(X_processed, y, problem_type, start_time)
        
        # Phase 5: Evaluation
        test_score = None
        if test_df is not None:
            test_score = self._evaluate_test(test_df, target_col)
        
        # Compile result
        total_time = time.time() - start_time
        
        if self.verbose:
            logger.info("=" * 70)
            logger.info("COMPLETE in %.1fs", total_time)
            if self.best_result:
                logger.info("Best: %s (Score: %.4f)", self.best_result['name'], self.best_result['score'])
            logger.info("=" * 70)
        
        return self._compile_result(test_score)
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Extract X, y and detect problem type."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target if categorical
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
        
        # Detect problem type
        n_unique = len(np.unique(y))
        if n_unique == 2:
            problem_type = 'binary'
        elif n_unique <= 20:
            problem_type = 'multiclass'
        else:
            problem_type = 'regression'
        
        if self.verbose:
            logger.info("Problem: %s (%d samples, %d features)", problem_type, len(X), len(X.columns))
        
        return X, y, problem_type
    
    def _preprocess(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Smart preprocessing."""
        if self.verbose:
            logger.info("Smart Preprocessing...")
        
        self.preprocessor = IntelligentPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y)
        
        if self.verbose:
            info = self.preprocessor.get_feature_info()
            logger.info("Original: %d features", len(X.columns))
            logger.info("Dropped: %d features", len(info['dropped']))
            logger.info("Final: %d features", len(X_processed.columns))
        
        # Create fingerprint
        self.fingerprint = self._create_fingerprint(X, y)
        
        return X_processed
    
    def _create_fingerprint(self, X: pd.DataFrame, y: np.ndarray) -> DatasetFingerprint:
        """Create dataset fingerprint."""
        # Calculate feature correlations
        X_num = X.select_dtypes(include=[np.number])
        if len(X_num.columns) > 1:
            corr_matrix = X_num.corr().abs()
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            corr_mean = np.mean(corr_values)
            corr_std = np.std(corr_values)
        else:
            corr_mean = 0
            corr_std = 0
        
        types = self.preprocessor.column_types if self.preprocessor else {'numerical': [], 'categorical_low': [], 'categorical_high': []}
        
        return DatasetFingerprint(
            n_samples=len(X),
            n_features=len(X.columns),
            n_categorical=len(types.get('categorical_low', [])) + len(types.get('categorical_high', [])),
            n_numerical=len(types.get('numerical', [])),
            sparsity=(X == 0).sum().sum() / X.size,
            missing_ratio=X.isnull().sum().sum() / X.size,
            target_cardinality=len(np.unique(y)),
            target_imbalance_ratio=self._compute_imbalance(y),
            estimated_noise=0.1,  # Simplified
            feature_corr_mean=corr_mean,
            feature_corr_std=corr_std
        )
    
    def _compute_imbalance(self, y: np.ndarray) -> float:
        """Compute class imbalance ratio."""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 2:
            return max(counts) / min(counts)
        return 1.0
    
    def _engineer_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Automatic feature engineering."""
        if self.verbose:
            logger.info("Feature Engineering...")
        
        self.feature_engineer = FeatureEngineer()
        X_engineered = self.feature_engineer.fit_transform(X, y)
        
        if self.verbose:
            logger.info("Created %d new features", len(X_engineered.columns) - len(X.columns))
        
        return X_engineered
    
    def _train_models(self, X: pd.DataFrame, y: np.ndarray, 
                     problem_type: str, start_time: float):
        """Train models with intelligence."""
        if self.verbose:
            logger.info("Training Models...")
        
        # Define model tiers (from simple to complex)
        model_tiers = self._get_model_tiers(problem_type)
        
        iteration = 0
        for tier_name, models in model_tiers.items():
            if time.time() - start_time > self.time_budget:
                if self.verbose:
                    logger.info("Time budget exceeded")
                break
            
            if self.verbose:
                logger.info("Tier: %s", tier_name)
            
            for model_name, model_class in models:
                if time.time() - start_time > self.time_budget:
                    break
                
                # Get hyperparameters (from meta-learning or defaults)
                params = self._get_hyperparameters(model_name)
                
                # Create and train model
                try:
                    model = model_class(**params)
                    result = self._evaluate_model(model, X, y, problem_type, model_name)
                    
                    self.all_results.append(result)
                    
                    if self.verbose:
                        logger.info("%-20s: %.4f (+/- %.4f)", model_name, result['score'], result['std'])
                    
                    # Check if we should stop
                    if self.adaptive and self.adaptive.should_stop(result['score'], iteration):
                        if self.verbose:
                            logger.info("Target achieved, stopping early")
                        # Select best before returning
                        if self.all_results:
                            self.best_result = max(self.all_results, key=lambda x: x['score'])
                            if self.metadb:
                                self._save_to_metadb()
                        return
                    
                    iteration += 1
                    
                except Exception as e:
                    if self.verbose:
                        logger.info("%-20s: FAILED (%s)", model_name, str(e)[:30])
                    continue
        
        # Select best
        if self.all_results:
            self.best_result = max(self.all_results, key=lambda x: x['score'])
            
            # Save to meta-learning DB
            if self.metadb:
                self._save_to_metadb()
    
    def _get_model_tiers(self, problem_type: str) -> Dict:
        """Get models organized by complexity."""
        # Binary/Multiclass classification
        if problem_type in ['binary', 'multiclass']:
            return {
                'fast': [
                    ('LogisticRegression', lambda **kw: LogisticRegression(**{**kw, 'max_iter': 1000})),
                    ('DecisionTree', lambda **kw: DecisionTreeClassifier(max_depth=5, **kw)),
                ],
                'balanced': [
                    ('RandomForest', lambda **kw: RandomForestClassifier(n_estimators=100, n_jobs=-1, **kw)),
                    ('XGBoost', lambda **kw: xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kw)),
                ],
                'accurate': [
                    ('XGBoost_Tuned', lambda **kw: xgb.XGBClassifier(
                        n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric='logloss', **kw
                    )),
                    ('LightGBM', lambda **kw: lgb.LGBMClassifier(n_estimators=200, verbose=-1, **kw)),
                ]
            }
        else:  # Regression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import Ridge, Lasso
            
            return {
                'fast': [
                    ('Ridge', Ridge),
                    ('Lasso', Lasso),
                ],
                'balanced': [
                    ('RandomForest', lambda **kw: RandomForestRegressor(n_estimators=100, n_jobs=-1, **kw)),
                ],
                'accurate': [
                    ('GradientBoosting', GradientBoostingRegressor),
                    ('XGBoost', lambda **kw: xgb.XGBRegressor(**kw)),
                ]
            }
    
    def _get_hyperparameters(self, model_name: str) -> Dict:
        """Get hyperparameters from meta-learning or defaults."""
        if self.hp_transfer and self.fingerprint:
            params = self.hp_transfer.get_initial_params(model_name, self.fingerprint)
            if params:
                return params
        
        # Default params
        return {}
    
    def _evaluate_model(self, model, X: pd.DataFrame, y: np.ndarray, 
                       problem_type: str, model_name: str) -> Dict:
        """Evaluate model with cross-validation."""
        start = time.time()
        
        # Cross-validation
        if problem_type == 'binary':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        elif problem_type == 'multiclass':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        else:  # regression
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            scores = -scores  # Convert to positive
        
        train_time = time.time() - start
        
        # Fit final model for feature importance
        model.fit(X, y)
        
        # Extract feature importance
        importance = self._get_feature_importance(model, X.columns)
        
        return {
            'name': model_name,
            'model': model,
            'score': scores.mean(),
            'std': scores.std(),
            'time': train_time,
            'importance': importance
        }
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance[0]
        else:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def _save_to_metadb(self):
        """Save experiment to meta-learning database."""
        for result in self.all_results:
            record = ExperimentRecord(
                fingerprint=self.fingerprint,
                problem_type='binary',  # Simplified
                model_name=result['name'],
                model_config={},  # Simplified
                cv_score=result['score'],
                cv_std=result['std'],
                train_time=result['time'],
                timestamp=datetime.now().isoformat()
            )
            self.metadb.add_experiment(record)
    
    def _evaluate_test(self, test_df: pd.DataFrame, target_col: str) -> float:
        """Evaluate on test set."""
        if self.best_result is None:
            return None
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Preprocess
        X_test_processed = self.preprocessor.transform(X_test)
        if self.feature_engineer:
            X_test_processed = self.feature_engineer.transform(X_test_processed)
        
        # Predict
        model = self.best_result['model']
        y_pred = model.predict(X_test_processed)
        
        # Score
        if len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test_processed)[:, 1]
            return roc_auc_score(y_test, y_proba)
        else:
            return accuracy_score(y_test, y_pred)
    
    def _compile_result(self, test_score: Optional[float]) -> AutoThinkResult:
        """Compile final result."""
        if not self.best_result:
            raise ValueError("No model was successfully trained")
        
        return AutoThinkResult(
            model=self.best_result['model'],
            model_name=self.best_result['name'],
            cv_score=self.best_result['score'],
            cv_std=self.best_result['std'],
            test_score=test_score,
            train_time=sum(r['time'] for r in self.all_results),
            preprocessor=self.preprocessor,
            feature_engineer=self.feature_engineer,
            fingerprint=self.fingerprint,
            all_results=self.all_results,
            feature_importance=self.best_result['importance']
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.best_result is None:
            raise ValueError("Must call fit() first")
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        if self.feature_engineer:
            X_processed = self.feature_engineer.transform(X_processed)
        
        return self.best_result['model'].predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if self.best_result is None:
            raise ValueError("Must call fit() first")
        
        X_processed = self.preprocessor.transform(X)
        if self.feature_engineer:
            X_processed = self.feature_engineer.transform(X_processed)
        
        return self.best_result['model'].predict_proba(X_processed)
