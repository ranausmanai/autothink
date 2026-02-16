"""
Smart Preprocessing - Phase 2
Intelligent handling of all data types
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder, OneHotEncoder
import warnings

warnings.filterwarnings('ignore')


class IntelligentPreprocessor(BaseEstimator, TransformerMixin):
    """
    Automatically preprocesses any dataframe intelligently.
    """
    
    def __init__(self, 
                 max_cardinality_onehot=10,
                 min_cardinality_target=100,
                 missing_threshold=0.5,
                 variance_threshold=0.0):
        self.max_cardinality_onehot = max_cardinality_onehot
        self.min_cardinality_target = min_cardinality_target
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        
        self.column_types = {}
        self.encoders = {}
        self.imputers = {}
        self.scalers = {}
        self.drop_columns = []
        self.feature_names = []
        
    def fit(self, X, y=None):
        X = X.copy()
        self.column_types = self._detect_column_types(X)
        self.drop_columns = self._identify_drop_columns(X)
        self._fit_imputers(X)
        if y is not None:
            self._fit_encoders(X, y)
        self._fit_scalers(X)
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.drop_columns, errors='ignore')
        X = self._apply_imputation(X)
        X = self._apply_encoding(X)
        X = self._apply_scaling(X)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.feature_names = list(X.columns)
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def _detect_column_types(self, X):
        types = {
            'numerical': [],
            'categorical_low': [],
            'categorical_high': [],
            'datetime': [],
            'text': [],
            'boolean': [],
            'constant': [],
            'id': []
        }
        
        for col in X.columns:
            series = X[col]
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            n_unique = series.nunique()
            
            if n_unique == len(series) and n_unique > 50:
                # Only flag as ID if it looks like an identifier (integer or string),
                # not a continuous numeric feature (float).
                if series.dtype == 'object' or (pd.api.types.is_integer_dtype(series)):
                    types['id'].append(col)
                    continue
            
            if n_unique == 1:
                types['constant'].append(col)
                continue
            
            if n_unique == 2:
                unique_vals = set(series.dropna().unique())
                if unique_vals <= {0, 1, True, False, 'yes', 'no', 'Y', 'N', 'true', 'false'}:
                    types['boolean'].append(col)
                    continue
            
            if pd.api.types.is_datetime64_any_dtype(series):
                types['datetime'].append(col)
                continue
            
            if series.dtype == 'object':
                try:
                    pd.to_datetime(series.dropna().iloc[:100] if len(series) > 0 else [])
                    types['datetime'].append(col)
                    continue
                except:
                    pass
            
            if pd.api.types.is_numeric_dtype(series):
                types['numerical'].append(col)
                continue
            
            if series.dtype == 'object':
                avg_len = series.astype(str).str.len().mean()
                if avg_len > 50 and unique_ratio > 0.5:
                    types['text'].append(col)
                    continue
            
            if n_unique <= self.max_cardinality_onehot:
                types['categorical_low'].append(col)
            else:
                types['categorical_high'].append(col)
        
        return types
    
    def _identify_drop_columns(self, X):
        drop_cols = []
        drop_cols.extend(self.column_types['constant'])
        drop_cols.extend(self.column_types['id'])
        
        for col in X.columns:
            if X[col].isnull().mean() > self.missing_threshold:
                drop_cols.append(col)
        
        return list(set(drop_cols))
    
    def _fit_imputers(self, X):
        num_cols = self.column_types['numerical']
        if num_cols:
            num_with_missing = [c for c in num_cols if X[c].isnull().any()]
            if num_with_missing:
                self.imputers['numerical'] = SimpleImputer(strategy='median')
                self.imputers['numerical'].fit(X[num_with_missing])
        
        cat_cols = (self.column_types['categorical_low'] + 
                   self.column_types['categorical_high'])
        if cat_cols:
            cat_with_missing = [c for c in cat_cols if X[c].isnull().any()]
            if cat_with_missing:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                self.imputers['categorical'].fit(X[cat_with_missing])
    
    def _fit_encoders(self, X, y):
        low_card_cols = self.column_types['categorical_low']
        if low_card_cols:
            self.encoders['onehot'] = OneHotEncoder(
                handle_unknown='ignore',
                use_cat_names=True
            )
            X_temp = X[low_card_cols].fillna('__MISSING__')
            self.encoders['onehot'].fit(X_temp)
        
        high_card_cols = self.column_types['categorical_high']
        if high_card_cols and y is not None:
            self.encoders['target'] = TargetEncoder(smoothing=1.0)
            X_temp = X[high_card_cols].fillna('__MISSING__')
            self.encoders['target'].fit(X_temp, y)
    
    def _fit_scalers(self, X):
        num_cols = self.column_types['numerical']
        if not num_cols:
            return
        
        X_num = X[num_cols].fillna(X[num_cols].median())
        
        outlier_ratios = []
        for col in num_cols:
            q1, q3 = X_num[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((X_num[col] < q1 - 3*iqr) | (X_num[col] > q3 + 3*iqr)).mean()
                outlier_ratios.append(outliers)
        
        avg_outliers = np.mean(outlier_ratios) if outlier_ratios else 0
        
        if avg_outliers > 0.05:
            self.scalers['numerical'] = RobustScaler()
        else:
            self.scalers['numerical'] = StandardScaler()
        
        self.scalers['numerical'].fit(X_num)
    
    def _apply_imputation(self, X):
        X = X.copy()
        if 'numerical' in self.imputers:
            num_cols = [c for c in self.column_types['numerical'] 
                       if c in X.columns and X[c].isnull().any()]
            if num_cols:
                X[num_cols] = self.imputers['numerical'].transform(X[num_cols])
        
        if 'categorical' in self.imputers:
            cat_cols = [c for c in (self.column_types['categorical_low'] + 
                                     self.column_types['categorical_high'])
                       if c in X.columns and X[c].isnull().any()]
            if cat_cols:
                X[cat_cols] = self.imputers['categorical'].transform(X[cat_cols])
        
        return X
    
    def _apply_encoding(self, X):
        X = X.copy()
        if 'onehot' in self.encoders:
            cols = [c for c in self.column_types['categorical_low'] if c in X.columns]
            if cols:
                X_temp = X[cols].fillna('__MISSING__')
                encoded = self.encoders['onehot'].transform(X_temp)
                X = pd.concat([X.drop(columns=cols), encoded], axis=1)
        
        if 'target' in self.encoders:
            cols = [c for c in self.column_types['categorical_high'] if c in X.columns]
            if cols:
                X_temp = X[cols].fillna('__MISSING__')
                encoded = self.encoders['target'].transform(X_temp)
                encoded.columns = [f'{c}_te' for c in encoded.columns]
                X = pd.concat([X.drop(columns=cols), encoded], axis=1)
        
        return X
    
    def _apply_scaling(self, X):
        if 'numerical' not in self.scalers:
            return X
        
        num_cols = [c for c in self.column_types['numerical'] if c in X.columns]
        if not num_cols:
            return X
        
        X = X.copy()
        X[num_cols] = self.scalers['numerical'].transform(X[num_cols])
        return X
    
    def get_feature_info(self):
        """Return information about features."""
        return {
            'dropped': self.drop_columns,
            'encoded_onehot': self.column_types.get('categorical_low', []),
            'encoded_target': self.column_types.get('categorical_high', []),
            'scaled': self.column_types.get('numerical', []),
            'final_features': self.feature_names
        }


class FeatureEngineer:
    """Automatic feature engineering."""
    
    def __init__(self, max_interactions=10, polynomial_degree=2):
        self.max_interactions = max_interactions
        self.polynomial_degree = polynomial_degree
        self.interaction_pairs = []
    
    def fit(self, X, y=None):
        if y is not None and len(X.columns) > 10:
            correlations = []
            for col in X.select_dtypes(include=[np.number]).columns:
                try:
                    corr = np.abs(np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [c for c, _ in correlations[:5]]
            
            self.interaction_pairs = [
                (top_features[i], top_features[j])
                for i in range(len(top_features))
                for j in range(i+1, len(top_features))
            ][:self.max_interactions]
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col1, col2 in self.interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
