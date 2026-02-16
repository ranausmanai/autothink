"""
FAST KAGGLE MODE - Balanced speed vs performance
"""

import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FastKaggleEnsemble:
    """Fast but powerful ensemble for Kaggle."""
    
    def __init__(self):
        self.models = []
        self.weights = None
        
    def fit(self, X, y):
        """Quick ensemble training."""
        logger.info("Training 5-model ensemble...")
        
        # Lightweight but diverse models
        self.models = [
            ('lgb1', lgb.LGBMClassifier(
                n_estimators=1000, num_leaves=31, learning_rate=0.05,
                feature_fraction=0.8, bagging_fraction=0.8,
                random_state=42, verbose=-1
            )),
            ('lgb2', lgb.LGBMClassifier(
                n_estimators=1000, num_leaves=63, learning_rate=0.03,
                feature_fraction=0.7, bagging_fraction=0.7,
                random_state=43, verbose=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgb3', lgb.LGBMClassifier(
                n_estimators=500, num_leaves=15, learning_rate=0.1,
                feature_fraction=0.9, bagging_fraction=0.9,
                random_state=44, verbose=-1
            )),
        ]
        
        # Train each
        oof_preds = np.zeros((len(X), len(self.models)))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.models):
                model.fit(X_tr, y_tr)
                if hasattr(model, 'predict_proba'):
                    oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
                else:
                    oof_preds[val_idx, i] = model.predict(X_val)
        
        # Optimize blending weights using CV
        logger.info("Optimizing blend weights...")
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            blend = np.dot(oof_preds, weights)
            from sklearn.metrics import roc_auc_score
            return -roc_auc_score(y, blend)
        
        # Initial equal weights
        x0 = [1.0 / len(self.models)] * len(self.models)
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds)
        self.weights = result.x / result.x.sum()
        
        logger.info("Optimized weights: %s", self.weights.round(3))
        
        # Retrain on full data
        for name, model in self.models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Weighted ensemble prediction."""
        preds = np.zeros((len(X), len(self.models)))
        
        for i, (name, model) in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                preds[:, i] = model.predict_proba(X)[:, 1]
            else:
                preds[:, i] = model.predict(X)
        
        # Weighted average
        return np.dot(preds, self.weights)


def create_kaggle_features(X, y=None):
    """Create competition-grade features quickly."""
    X_new = X.copy()
    
    # 1. Log transforms
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].min() >= 0 and X[col].max() > 0:
            X_new[f'{col}_log'] = np.log1p(X[col])
        X_new[f'{col}_sq'] = X[col] ** 2
    
    # 2. Ratios between numeric features (top 10)
    num_cols = X.select_dtypes(include=[np.number]).columns[:10]
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:min(i+4, len(num_cols))]:  # Limit combinations
            X_new[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
    
    # 3. Statistics by categorical (if exists)
    cat_cols = X.select_dtypes(include=['object']).columns
    for cat_col in cat_cols[:2]:  # Top 2 categoricals
        for num_col in num_cols[:3]:
            group_mean = X.groupby(cat_col)[num_col].transform('mean')
            X_new[f'{num_col}_by_{cat_col}'] = group_mean
    
    X_new = X_new.fillna(0)
    
    # Select best features if too many
    if y is not None and len(X_new.columns) > 100:
        selector = SelectKBest(mutual_info_classif, k=100)
        X_selected = selector.fit_transform(X_new, y)
        selected_cols = X_new.columns[selector.get_support()]
        X_new = pd.DataFrame(X_selected, columns=selected_cols, index=X_new.index)
    
    return X_new


def fit_kaggle_fast(df, target_col, time_budget=300):
    """Fast Kaggle mode - 5-10 minutes for good results."""
    logger.info("=" * 70)
    logger.info("KAGGLE FAST MODE - Optimized for Leaderboard")
    logger.info("=" * 70)
    
    import time
    start = time.time()
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    
    # Drop IDs
    id_cols = [c for c in X.columns if 'id' in c.lower()]
    X = X.drop(columns=id_cols, errors='ignore')
    
    logger.info("Data: %s samples, %d features", f"{len(X):,}", X.shape[1])
    
    # Feature engineering
    logger.info("Creating Kaggle features...")
    X_feat = create_kaggle_features(X, y)
    logger.info("Features: %d -> %d", X.shape[1], X_feat.shape[1])
    
    # Train ensemble
    logger.info("Training optimized ensemble...")
    ensemble = FastKaggleEnsemble()
    ensemble.fit(X_feat, y)
    
    elapsed = time.time() - start
    logger.info("Complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
    
    return ensemble, X_feat.columns
