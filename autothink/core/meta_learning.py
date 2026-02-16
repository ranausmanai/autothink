"""
Meta-Learning Database - Phase 3
Learn from past experiences to make better decisions
"""

import logging
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DatasetFingerprint:
    """Compact representation of dataset characteristics."""
    n_samples: int
    n_features: int
    n_categorical: int
    n_numerical: int
    sparsity: float
    missing_ratio: float
    target_cardinality: int
    target_imbalance_ratio: float
    estimated_noise: float
    feature_corr_mean: float
    feature_corr_std: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for similarity search."""
        return np.array([
            np.log1p(self.n_samples),
            np.log1p(self.n_features),
            self.n_categorical / max(self.n_features, 1),
            self.n_numerical / max(self.n_features, 1),
            self.sparsity,
            self.missing_ratio,
            np.log1p(self.target_cardinality),
            np.log1p(self.target_imbalance_ratio),
            self.estimated_noise,
            self.feature_corr_mean,
            self.feature_corr_std
        ])


@dataclass
class ExperimentRecord:
    """Single experiment result."""
    fingerprint: DatasetFingerprint
    problem_type: str
    model_name: str
    model_config: Dict
    cv_score: float
    cv_std: float
    train_time: float
    timestamp: str
    dataset_name: Optional[str] = None


class MetaLearningDB:
    """
    Database of past experiments for transfer learning.
    
    Enables:
    1. Finding similar past datasets
    2. Recommending hyperparameters
    3. Estimating expected performance
    4. Ranking model choices
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = "~/.autothink/metadb.pkl"
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.experiments: List[ExperimentRecord] = []
        self.dataset_index: Dict[str, List[int]] = defaultdict(list)
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        
        self._load()
    
    def _load(self):
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.experiments = data.get('experiments', [])
                    self._rebuild_indices()
            except Exception as e:
                logger.warning("Could not load meta-learning DB: %s", e)
                self.experiments = []
    
    def _save(self):
        """Save database to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'experiments': self.experiments,
                    'version': '1.0'
                }, f)
        except Exception as e:
            logger.warning("Could not save meta-learning DB: %s", e)
    
    def _rebuild_indices(self):
        """Rebuild indices after loading."""
        self.dataset_index = defaultdict(list)
        self.model_performance = defaultdict(list)
        
        for i, exp in enumerate(self.experiments):
            if exp.dataset_name:
                self.dataset_index[exp.dataset_name].append(i)
            self.model_performance[exp.model_name].append(exp.cv_score)
    
    def add_experiment(self, record: ExperimentRecord):
        """Add new experiment to database."""
        self.experiments.append(record)
        
        # Update indices
        idx = len(self.experiments) - 1
        if record.dataset_name:
            self.dataset_index[record.dataset_name].append(idx)
        self.model_performance[record.model_name].append(record.cv_score)
        
        # Save periodically
        if len(self.experiments) % 10 == 0:
            self._save()
    
    def find_similar(self, fingerprint: DatasetFingerprint, k: int = 5) -> List[ExperimentRecord]:
        """Find k most similar past experiments."""
        if not self.experiments:
            return []
        
        # Convert fingerprints to vectors
        target_vec = fingerprint.to_vector()
        
        # Calculate similarities
        similarities = []
        for i, exp in enumerate(self.experiments):
            exp_vec = exp.fingerprint.to_vector()
            # Cosine similarity
            sim = np.dot(target_vec, exp_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(exp_vec) + 1e-8)
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        top_indices = [i for i, _ in similarities[:k]]
        return [self.experiments[i] for i in top_indices]
    
    def recommend_hyperparameters(self, model_name: str, 
                                   fingerprint: DatasetFingerprint) -> Optional[Dict]:
        """Recommend hyperparameters based on similar datasets."""
        similar = self.find_similar(fingerprint, k=10)
        
        # Filter for same model
        similar_same_model = [s for s in similar if s.model_name == model_name]
        
        if not similar_same_model:
            return None
        
        # Get best config from similar
        best = max(similar_same_model, key=lambda x: x.cv_score)
        return best.model_config
    
    def estimate_performance(self, model_name: str, 
                            fingerprint: DatasetFingerprint) -> tuple:
        """Estimate expected performance for a model on a dataset."""
        similar = self.find_similar(fingerprint, k=20)
        
        # Get scores for this model on similar datasets
        scores = [s.cv_score for s in similar if s.model_name == model_name]
        
        if not scores:
            # Use global average
            scores = self.model_performance.get(model_name, [0.5])
        
        return np.mean(scores), np.std(scores)
    
    def rank_models(self, fingerprint: DatasetFingerprint, 
                   candidate_models: List[str]) -> List[tuple]:
        """Rank candidate models by expected performance."""
        rankings = []
        
        for model_name in candidate_models:
            mean_score, std_score = self.estimate_performance(model_name, fingerprint)
            # Score minus uncertainty (conservative estimate)
            conservative_score = mean_score - 0.5 * std_score
            rankings.append((model_name, conservative_score, mean_score, std_score))
        
        # Sort by conservative score
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'n_experiments': len(self.experiments),
            'n_unique_datasets': len(self.dataset_index),
            'models_tested': list(self.model_performance.keys()),
            'avg_scores': {k: np.mean(v) for k, v in self.model_performance.items()}
        }


class AdaptivePipeline:
    """
    Adaptive pipeline that adjusts based on intermediate results.
    
    Strategy:
    1. Try simplest model first
    2. If good enough, stop
    3. If not, escalate complexity
    4. Use meta-learning to guide escalation
    """
    
    def __init__(self, metadb: MetaLearningDB, target_score: float = 0.95):
        self.metadb = metadb
        self.target_score = target_score
        self.results = []
        
    def should_stop(self, current_score: float, iteration: int) -> bool:
        """Determine if we should stop trying more models."""
        # Stop if we hit target
        if current_score >= self.target_score:
            return True
        
        # Stop if we've tried enough
        if iteration >= 10:
            return True
        
        # Stop if diminishing returns (last 3 models didn't improve much)
        if len(self.results) >= 3:
            recent_scores = [r.cv_score for r in self.results[-3:]]
            if max(recent_scores) - min(recent_scores) < 0.01:
                return True
        
        return False
    
    def get_next_model_complexity(self, current_tier: str) -> str:
        """Escalate model complexity."""
        tiers = ['fast', 'balanced', 'accurate', 'ensemble']
        current_idx = tiers.index(current_tier) if current_tier in tiers else 0
        next_idx = min(current_idx + 1, len(tiers) - 1)
        return tiers[next_idx]
    
    def log_result(self, result):
        """Log a result for decision making."""
        self.results.append(result)


class HyperparameterTransfer:
    """
    Transfer hyperparameters from similar datasets.
    """
    
    def __init__(self, metadb: MetaLearningDB):
        self.metadb = metadb
        
    def get_initial_params(self, model_name: str, 
                          fingerprint: DatasetFingerprint) -> Dict:
        """Get good starting hyperparameters."""
        # Try to get from similar datasets
        config = self.metadb.recommend_hyperparameters(model_name, fingerprint)
        
        if config:
            return config
        
        # Fall back to smart defaults
        return self._get_default_params(model_name, fingerprint)
    
    def _get_default_params(self, model_name: str, 
                           fingerprint: DatasetFingerprint) -> Dict:
        """Get smart defaults based on dataset characteristics."""
        defaults = {
            'XGBClassifier': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            },
            'LGBMClassifier': {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            },
            'RandomForestClassifier': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
            },
            'LogisticRegression': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
            }
        }
        
        params = defaults.get(model_name, {}).copy()
        
        # Adapt to dataset
        if fingerprint.n_samples < 1000:
            # Small data: Regularize more
            if 'reg_lambda' in params:
                params['reg_lambda'] = 10.0
            if 'reg_alpha' in params:
                params['reg_alpha'] = 1.0
            if 'max_depth' in params:
                params['max_depth'] = 3
        
        if fingerprint.target_imbalance_ratio > 10:
            # Imbalanced: Adjust for class weights
            if model_name in ['XGBClassifier', 'LGBMClassifier']:
                params['scale_pos_weight'] = fingerprint.target_imbalance_ratio
        
        return params
