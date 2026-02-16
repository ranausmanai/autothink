"""
IMPROVED FEATURE ENGINEERING (DEPRECATED)

This module contains domain-specific (medical) feature engineering with
hardcoded thresholds. It is kept for backward compatibility but should not
be used for new projects. Use ``feature_engineering_general.py`` instead,
which learns thresholds from data automatically.
"""

import logging
import warnings as _warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

_warnings.warn(
    "autothink.core.feature_engineering_v2 is deprecated. "
    "Use autothink.core.feature_engineering_general instead.",
    DeprecationWarning,
    stacklevel=2,
)


class SmartFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    ACTUALLY SMART feature engineering.
    
    Problems with old version:
    1. Didn't create Age groups (50+, 60+)
    2. Didn't detect BP thresholds (140, 160)
    3. Didn't create medical risk scores
    
    This version fixes all that.
    """
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.created_features = []
        
    def fit(self, X, y=None):
        """Learn what features to create."""
        self.created_features = []
        
        # Detect column types
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return self
    
    def transform(self, X):
        """Create features that ACTUALLY matter."""
        X_new = X.copy()
        
        # ============================================================
        # MEDICAL / DOMAIN-SPECIFIC FEATURES
        # ============================================================
        
        # Age-based features (cardiovascular risk)
        age_cols = [c for c in self.numeric_cols if 'age' in c.lower()]
        for col in age_cols:
            X_new[f'{col}_gte50'] = (X[col] >= 50).astype(int)
            X_new[f'{col}_gte60'] = (X[col] >= 60).astype(int)
            X_new[f'{col}_gte70'] = (X[col] >= 70).astype(int)
            self.created_features.extend([f'{col}_gte50', f'{col}_gte60', f'{col}_gte70'])
        
        # Blood pressure features (hypertension thresholds)
        bp_cols = [c for c in self.numeric_cols if 'bp' in c.lower() or 'blood' in c.lower() or 'pressure' in c.lower()]
        for col in bp_cols:
            X_new[f'{col}_gte120'] = (X[col] >= 120).astype(int)
            X_new[f'{col}_gte140'] = (X[col] >= 140).astype(int)
            X_new[f'{col}_gte160'] = (X[col] >= 160).astype(int)
            self.created_features.extend([f'{col}_gte120', f'{col}_gte140', f'{col}_gte160'])
        
        # Cholesterol features
        chol_cols = [c for c in self.numeric_cols if 'chol' in c.lower() or 'cholesterol' in c.lower()]
        for col in chol_cols:
            X_new[f'{col}_gte200'] = (X[col] >= 200).astype(int)
            X_new[f'{col}_gte240'] = (X[col] >= 240).astype(int)
            self.created_features.extend([f'{col}_gte200', f'{col}_gte240'])
        
        # Heart rate features (220 - age formula)
        hr_cols = [c for c in self.numeric_cols if 'hr' in c.lower() or 'heart' in c.lower() or 'max' in c.lower()]
        age_col = [c for c in self.numeric_cols if 'age' in c.lower()][0] if age_cols else None
        
        if hr_cols and age_col:
            for hr_col in hr_cols[:1]:  # First HR column
                X_new[f'{hr_col}_Predicted'] = 220 - X[age_col]
                X_new[f'{hr_col}_Reserve'] = X_new[f'{hr_col}_Predicted'] - X[hr_col]
                X_new[f'{hr_col}_Ratio'] = X[hr_col] / X_new[f'{hr_col}_Predicted']
                self.created_features.extend([f'{hr_col}_Predicted', f'{hr_col}_Reserve', f'{hr_col}_Ratio'])
        
        # ============================================================
        # RISK SCORES (The key to winning!)
        # ============================================================
        
        # Create a master risk score if we have the right columns
        risk_components = []
        
        if age_cols:
            age_col = age_cols[0]
            X_new['Age_Risk'] = (X[age_col] >= 60).astype(int) * 0.3 + (X[age_col] >= 50).astype(int) * 0.2
            risk_components.append('Age_Risk')
        
        if bp_cols:
            bp_col = bp_cols[0]
            X_new['BP_Risk'] = (X[bp_col] >= 160).astype(int) * 0.25 + (X[bp_col] >= 140).astype(int) * 0.15
            risk_components.append('BP_Risk')
        
        if chol_cols:
            chol_col = chol_cols[0]
            X_new['Chol_Risk'] = (X[chol_col] >= 240).astype(int) * 0.15
            risk_components.append('Chol_Risk')
        
        # ST depression (if exists)
        st_cols = [c for c in self.numeric_cols if 'st' in c.lower() or 'depression' in c.lower()]
        for col in st_cols:
            X_new[f'{col}_gt1'] = (X[col] > 1).astype(int)
            X_new[f'{col}_gt2'] = (X[col] > 2).astype(int)
            X_new['ST_Risk'] = X_new[f'{col}_gt1'] * 0.2 + X_new[f'{col}_gt2'] * 0.3
            risk_components.append('ST_Risk')
            self.created_features.extend([f'{col}_gt1', f'{col}_gt2', 'ST_Risk'])
        
        # Exercise angina (if exists)
        angina_cols = [c for c in self.numeric_cols if 'angina' in c.lower() or 'exercise' in c.lower()]
        for col in angina_cols:
            X_new['Angina_Risk'] = X[col] * 0.25
            risk_components.append('Angina_Risk')
        
        # COMBINED RISK SCORE
        if risk_components:
            X_new['Risk_Master'] = sum(X_new[col] for col in risk_components if col in X_new.columns)
            self.created_features.append('Risk_Master')
        
        # ============================================================
        # INTERACTIONS (ratios and products)
        # ============================================================
        
        # Cholesterol per Age
        if chol_cols and age_cols:
            X_new['Chol_per_Age'] = X[chol_cols[0]] / (X[age_cols[0]] + 1)
            self.created_features.append('Chol_per_Age')
        
        # BP per Age
        if bp_cols and age_cols:
            X_new['BP_per_Age'] = X[bp_cols[0]] / (X[age_cols[0]] + 1)
            self.created_features.append('BP_per_Age')
        
        # Age x BP interaction
        if bp_cols and age_cols:
            X_new['Age_x_BP'] = X[age_cols[0]] * X[bp_cols[0]] / 1000
            self.created_features.append('Age_x_BP')
        
        # ============================================================
        # MATHEMATICAL TRANSFORMS
        # ============================================================
        
        for col in self.numeric_cols[:5]:  # Top 5 numeric only
            # Log transform (for skewed data)
            if X[col].min() >= 0:
                X_new[f'{col}_log'] = np.log1p(X[col])
                self.created_features.append(f'{col}_log')
            
            # Square
            X_new[f'{col}_sq'] = X[col] ** 2
            self.created_features.append(f'{col}_sq')
        
        # Fill NaNs
        X_new = X_new.fillna(0)
        
        return X_new
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class DomainAwareEngineer(BaseEstimator, TransformerMixin):
    """
    Automatically detects domain (medical, financial, etc.) and engineers appropriate features.
    """
    
    def __init__(self):
        self.domain = 'generic'
        self.engineer = None
    
    def detect_domain(self, X, y=None):
        """Detect what domain the data is from."""
        columns = [c.lower() for c in X.columns]
        
        # Medical indicators
        medical_terms = ['age', 'bp', 'blood', 'pressure', 'chol', 'cholesterol', 
                        'heart', 'hr', 'rate', 'st', 'depression', 'angina', 
                        'vessel', 'chest', 'pain', 'patient', 'medical', 'disease']
        medical_score = sum(1 for term in medical_terms if any(term in c for c in columns))
        
        # Financial indicators
        financial_terms = ['amount', 'balance', 'income', 'salary', 'price', 'cost', 
                          'revenue', 'profit', 'loan', 'credit', 'debt', 'payment']
        financial_score = sum(1 for term in financial_terms if any(term in c for c in columns))
        
        if medical_score >= 3:
            return 'medical'
        elif financial_score >= 3:
            return 'financial'
        else:
            return 'generic'
    
    def fit(self, X, y=None):
        self.domain = self.detect_domain(X, y)
        logger.info("Detected domain: %s", self.domain)
        
        if self.domain == 'medical':
            self.engineer = SmartFeatureEngineer()
        else:
            self.engineer = SmartFeatureEngineer()  # Default for now
        
        self.engineer.fit(X, y)
        return self
    
    def transform(self, X):
        return self.engineer.transform(X)
