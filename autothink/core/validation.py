"""
Validation & Safety Checks
Prevent data leakage and overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


class LeakageDetector:
    """
    Detect and prevent data leakage.
    
    Types of leakage:
    1. ID leakage: Column with unique values per row
    2. Target leakage: Feature that wouldn't be available at prediction time
    3. Temporal leakage: Future information in past data
    """
    
    @staticmethod
    def check_id_leakage(X: pd.DataFrame, threshold: float = 0.999) -> List[str]:
        """
        Detect columns that are likely IDs (unique per row).
        More conservative - only flag if almost certainly an ID.
        """
        id_candidates = []
        
        for col in X.columns:
            n_unique = X[col].nunique()
            n_total = len(X)
            unique_ratio = n_unique / n_total
            
            # Only if 100% unique (or very close) AND named like ID
            is_named_like_id = any(keyword in col.lower() for keyword in ['id', 'key', 'index', 'seq', 'number'])
            is_100_percent_unique = unique_ratio >= threshold
            
            if is_100_percent_unique and is_named_like_id:
                id_candidates.append({
                    'column': col,
                    'unique_ratio': unique_ratio,
                    'reason': 'Named like ID and 100% unique'
                })
        
        return id_candidates
    
    @staticmethod
    def check_target_leakage(X: pd.DataFrame, y: pd.Series, 
                            threshold: float = 0.95) -> List[Dict]:
        """
        Detect features that might be leaking target information.
        
        Returns:
            List of suspicious columns
        """
        suspicious = []
        
        for col in X.select_dtypes(include=[np.number]).columns:
            # Check correlation with target
            correlation = np.abs(X[col].corr(y))
            
            if correlation > threshold:
                suspicious.append({
                    'column': col,
                    'correlation': correlation,
                    'reason': f'Correlation with target: {correlation:.3f}'
                })
        
        return suspicious
    
    @staticmethod
    def validate_no_leakage(X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        Comprehensive leakage check.
        
        Returns:
            Dictionary with leakage report
        """
        report = {
            'has_leakage': False,
            'warnings': [],
            'critical': [],
            'columns_to_drop': []
        }
        
        # Check for ID leakage
        id_leaks = LeakageDetector.check_id_leakage(X)
        for leak in id_leaks:
            if leak['unique_ratio'] > 0.999:  # Almost certainly an ID
                report['critical'].append(
                    f"CRITICAL: '{leak['column']}' appears to be an ID "
                    f"({leak['unique_ratio']:.1%} unique). This will cause overfitting!"
                )
                report['columns_to_drop'].append(leak['column'])
                report['has_leakage'] = True
            else:
                report['warnings'].append(
                    f"Warning: '{leak['column']}' might be an ID "
                    f"({leak['unique_ratio']:.1%} unique)"
                )
        
        # Check for target leakage
        if y is not None:
            target_leaks = LeakageDetector.check_target_leakage(X, y)
            for leak in target_leaks:
                report['critical'].append(
                    f"CRITICAL: '{leak['column']}' highly correlated with target "
                    f"({leak['correlation']:.3f}). Possible data leakage!"
                )
                report['columns_to_drop'].append(leak['column'])
                report['has_leakage'] = True
        
        return report


class OverfittingDetector:
    """
    Detect and prevent overfitting.
    """
    
    @staticmethod
    def check_cv_sanity(cv_scores: np.ndarray, threshold: float = 0.95) -> Dict:
        """
        Check if CV scores are suspiciously high.
        
        Returns:
            Dictionary with assessment
        """
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        issues = []
        
        if mean_score > threshold:
            issues.append(
                f"CV score ({mean_score:.3f}) suspiciously high. "
                "Possible data leakage or overfitting."
            )
        
        if std_score < 0.001:
            issues.append(
                f"CV std ({std_score:.5f}) suspiciously low. "
                "Model may be memorizing data."
            )
        
        return {
            'is_suspicious': len(issues) > 0,
            'mean_score': mean_score,
            'std_score': std_score,
            'issues': issues
        }
    
    @staticmethod
    def recommend_model_complexity(n_samples: int, n_features: int) -> Dict:
        """
        Recommend model complexity based on data size.
        
        Returns:
            Dictionary with recommendations
        """
        ratio = n_samples / max(n_features, 1)
        
        if n_samples < 500:
            return {
                'complexity': 'very_low',
                'max_depth': 3,
                'min_samples_leaf': 10,
                'regularization': 'high',
                'recommendation': 'Use simple linear models only. Aggressive regularization required.'
            }
        elif n_samples < 1000:
            return {
                'complexity': 'low',
                'max_depth': 4,
                'min_samples_leaf': 5,
                'regularization': 'medium_high',
                'recommendation': 'Limit tree depth. Use regularization.'
            }
        elif ratio < 10:
            return {
                'complexity': 'medium',
                'max_depth': 5,
                'min_samples_leaf': 3,
                'regularization': 'medium',
                'recommendation': 'More features than ideal. Consider feature selection.'
            }
        else:
            return {
                'complexity': 'high',
                'max_depth': 6,
                'min_samples_leaf': 2,
                'regularization': 'low',
                'recommendation': 'Good sample-to-feature ratio. Can use complex models.'
            }


class DataValidator:
    """
    Comprehensive data validation before modeling.
    """
    
    @staticmethod
    def validate_dataset(X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        Run all validation checks.
        
        Returns:
            Validation report
        """
        report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check 1: Empty data
        if X.empty:
            report['errors'].append("Dataset is empty")
            report['is_valid'] = False
            return report
        
        # Check 2: Too few samples
        if len(X) < 10:
            report['errors'].append(f"Too few samples: {len(X)} (minimum 10)")
            report['is_valid'] = False
        elif len(X) < 100:
            report['warnings'].append(f"Very small dataset: {len(X)} samples")
            report['recommendations'].append("Use simple models with strong regularization")
        
        # Check 3: Too many features
        n_features = X.shape[1]
        if n_features > len(X) / 2:
            report['warnings'].append(
                f"Many features ({n_features}) relative to samples ({len(X)})"
            )
            report['recommendations'].append("Consider feature selection or dimensionality reduction")
        
        # Check 4: Constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            report['warnings'].append(f"Constant features (no information): {constant_features}")
        
        # Check 5: Missing values
        missing_pct = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        if missing_pct > 0.5:
            report['warnings'].append(f"Dataset is {missing_pct:.1%} missing values")
        
        # Check 6: Leakage
        leakage_report = LeakageDetector.validate_no_leakage(X, y)
        if leakage_report['has_leakage']:
            report['errors'].extend(leakage_report['critical'])
            report['recommendations'].append(
                f"Drop these columns to prevent leakage: {leakage_report['columns_to_drop']}"
            )
            report['is_valid'] = False
        
        report['warnings'].extend(leakage_report['warnings'])
        
        # Check 7: Target issues
        if y is not None:
            if y.nunique() < 2:
                report['errors'].append("Target has only one class")
                report['is_valid'] = False
            
            # Check imbalance
            if y.nunique() == 2:
                class_counts = y.value_counts()
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 100:
                    report['warnings'].append(f"Severe class imbalance: {imbalance_ratio:.0f}:1")
                    report['recommendations'].append("Use class weights, stratified sampling, or anomaly detection")
        
        return report


class PostTrainingValidator:
    """
    Validate model after training.
    """
    
    @staticmethod
    def validate_model_performance(train_score: float, 
                                   val_score: float,
                                   test_score: float = None) -> Dict:
        """
        Check for overfitting post-training.
        
        Returns:
            Validation report
        """
        report = {
            'is_valid': True,
            'warnings': [],
            'overfitting_detected': False
        }
        
        # Check train vs validation gap
        gap = train_score - val_score
        if gap > 0.1:  # More than 10% drop
            report['overfitting_detected'] = True
            report['warnings'].append(
                f"Overfitting detected: train={train_score:.3f}, val={val_score:.3f} (gap={gap:.3f})"
            )
            report['recommendations'] = [
                "Increase regularization",
                "Reduce model complexity",
                "Get more training data",
                "Use feature selection"
            ]
        
        # Check if validation is suspiciously high
        if val_score > 0.99:
            report['warnings'].append(
                f"Validation score ({val_score:.3f}) suspiciously high. Check for data leakage."
            )
        
        # Check test score if available
        if test_score is not None:
            test_gap = val_score - test_score
            if test_gap > 0.05:
                report['warnings'].append(
                    f"Test performance ({test_score:.3f}) significantly lower than validation ({val_score:.3f})"
                )
        
        return report
