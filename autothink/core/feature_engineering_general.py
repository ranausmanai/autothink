"""
GENERAL FEATURE ENGINEERING - Works on ANY tabular data

Key insight: Instead of hardcoding "Age >= 50", 
let the DATA tell us what thresholds matter.
"""

import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GeneralFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    ACTUALLY GENERAL feature engineering.
    
    Works on any data by LEARNING patterns from the data itself,
    not hardcoding domain knowledge.
    """
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.thresholds = {}  # Learned thresholds for each column
        self.interactions = []  # Learned interaction pairs
        
    def find_best_threshold(self, X_col, y, n_thresholds=10):
        """
        Find the threshold on X_col that best separates y.

        Instead of hardcoding Age >= 50, we learn from data
        that Age >= 47.3 is actually the best split.
        Uses Classifier for classification targets, Regressor for regression.
        """
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Pick tree type based on target
        if hasattr(y, 'dtype') and y.dtype.kind == 'f' and y.nunique() > 20:
            tree = DecisionTreeRegressor(max_depth=1, random_state=42)
        else:
            tree = DecisionTreeClassifier(max_depth=1, random_state=42)

        tree.fit(X_col.values.reshape(-1, 1), y)

        # Extract threshold
        if hasattr(tree.tree_, 'threshold'):
            threshold = tree.tree_.threshold[0]
            if threshold != -2:  # -2 means no split found
                return threshold

        # Fallback: use percentile-based thresholds
        return np.percentile(X_col, [25, 50, 75])
    
    def fit(self, X, y):
        """Learn what features to create from the DATA."""
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Learn optimal thresholds for each numeric column
        logger.info("Learning optimal thresholds from data...")
        for col in self.numeric_cols:
            # Skip if too few unique values (already categorical)
            if X[col].nunique() < 10:
                continue
                
            # Find best threshold
            try:
                thresh = self.find_best_threshold(X[col], y)
                if isinstance(thresh, np.ndarray):
                    self.thresholds[col] = thresh.tolist()
                else:
                    self.thresholds[col] = [thresh]
            except:
                # If tree fails, use percentiles
                self.thresholds[col] = [
                    X[col].quantile(0.25),
                    X[col].quantile(0.5),
                    X[col].quantile(0.75)
                ]
        
        # Learn interactions (correlation-based)
        logger.info("Learning feature interactions...")
        if len(self.numeric_cols) >= 2:
            # Find pairs with high mutual information
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            # Sample for speed
            sample_size = min(10000, len(X))
            X_sample = X[self.numeric_cols].sample(n=sample_size, random_state=42)
            y_sample = y.sample(n=sample_size, random_state=42)

            # Pick MI function based on target type
            if hasattr(y, 'dtype') and y.dtype.kind == 'f' and y.nunique() > 20:
                mi_func = mutual_info_regression
            else:
                mi_func = mutual_info_classif

            # Calculate MI for each feature
            mi_scores = mi_func(X_sample.fillna(0), y_sample, random_state=42)
            top_features = [self.numeric_cols[i] for i in np.argsort(mi_scores)[-5:]]  # Top 5
            
            # Create interactions among top features
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    self.interactions.append((col1, col2))
        
        return self
    
    def transform(self, X):
        """Create features based on LEARNED patterns."""
        X_new = X.copy()
        
        # ============================================================
        # LEARNED THRESHOLDS (Not hardcoded!)
        # ============================================================
        
        for col, thresholds in self.thresholds.items():
            for i, thresh in enumerate(thresholds):
                if thresh > X[col].min() and thresh < X[col].max():  # Valid threshold
                    # Clean column name (remove special chars)
                    clean_col = col.replace(' ', '_').replace('-', '_')
                    X_new[f'{clean_col}_gte_{int(thresh*10)}'] = (X[col] >= thresh).astype(int)
        
        # ============================================================
        # LEARNED INTERACTIONS
        # ============================================================
        
        for col1, col2 in self.interactions[:10]:  # Limit to avoid explosion
            # Clean names
            c1 = col1.replace(' ', '_').replace('-', '_')[:20]
            c2 = col2.replace(' ', '_').replace('-', '_')[:20]
            # Ratio
            X_new[f'{c1}_div_{c2}'] = X[col1] / (X[col2] + 1e-8)
            # Product
            X_new[f'{c1}_mul_{c2}'] = X[col1] * X[col2]
            # Difference
            X_new[f'{c1}_minus_{c2}'] = X[col1] - X[col2]
        
        # ============================================================
        # AUTOMATIC RISK SCORES
        # ============================================================
        
        # Create a "risk score" from top binary features
        binary_cols = [c for c in X_new.columns if '_gte_' in c]
        if len(binary_cols) >= 3:
            # Simple weighted sum of risk indicators
            X_new['Auto_Risk_Score'] = X_new[binary_cols[:5]].sum(axis=1) / len(binary_cols[:5])
        
        # ============================================================
        # MATHEMATICAL TRANSFORMS
        # ============================================================
        
        for col in self.numeric_cols[:5]:  # Top 5 numeric
            clean_col = col.replace(' ', '_').replace('-', '_')[:20]
            # Log (for skewed distributions)
            if X[col].min() >= 0 and X[col].skew() > 1:
                X_new[f'{clean_col}_log'] = np.log1p(X[col])
            
            # Square (captures non-linearity)
            X_new[f'{clean_col}_sq'] = X[col] ** 2
            
            # Square root
            if X[col].min() >= 0:
                X_new[f'{clean_col}_sqrt'] = np.sqrt(X[col])
        
        # Fill NaNs
        X_new = X_new.fillna(0)
        
        return X_new
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class AdaptiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adapts feature engineering based on data size and complexity.
    """
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.engineer = None
        
    def fit(self, X, y):
        """Choose appropriate engineering strategy."""
        n_samples = len(X)
        n_features = X.shape[1]
        
        if n_samples < 1000:
            # Small data: be conservative
            logger.info("Small dataset: Conservative feature engineering")
            self.engineer = GeneralFeatureEngineer(max_features=50)
        elif n_samples < 10000:
            # Medium data: moderate
            logger.info("Medium dataset: Standard feature engineering")
            self.engineer = GeneralFeatureEngineer(max_features=100)
        else:
            # Large data: aggressive
            logger.info("Large dataset: Aggressive feature engineering")
            self.engineer = GeneralFeatureEngineer(max_features=200)
        
        self.engineer.fit(X, y)
        return self
    
    def transform(self, X):
        return self.engineer.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
