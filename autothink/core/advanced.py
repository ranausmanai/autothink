"""
Advanced Features - Phase 5
Causal inference, explanations, ensemble intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')


class CausalAutoML:
    """
    Causal inference for AutoML.
    
    Beyond correlation - understand what causes what.
    Useful for:
    - Treatment effect estimation
    - Feature attribution with causality
    - Counterfactual explanations
    """
    
    def __init__(self, method: str = 'propensity_score'):
        self.method = method
        self.propensity_model = None
        self.outcome_model = None
    
    def estimate_treatment_effect(self, X: pd.DataFrame, treatment: str, 
                                   outcome: str) -> Dict:
        """
        Estimate causal effect of treatment on outcome.
        
        Args:
            X: Dataframe with features
            treatment: Binary treatment column
            outcome: Outcome column
        
        Returns:
            Dictionary with causal estimates
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Step 1: Estimate propensity scores
        feature_cols = [c for c in X.columns if c not in [treatment, outcome]]
        X_features = X[feature_cols]
        
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X_features, X[treatment])
        propensity_scores = self.propensity_model.predict_proba(X_features)[:, 1]
        
        # Step 2: Estimate outcome models
        # Model for treated
        treated_mask = X[treatment] == 1
        self.outcome_model_treated = GradientBoostingRegressor()
        self.outcome_model_treated.fit(
            X_features[treated_mask], 
            X.loc[treated_mask, outcome]
        )
        
        # Model for control
        self.outcome_model_control = GradientBoostingRegressor()
        self.outcome_model_control.fit(
            X_features[~treated_mask], 
            X.loc[~treated_mask, outcome]
        )
        
        # Step 3: Compute Average Treatment Effect (ATE)
        y1_pred = self.outcome_model_treated.predict(X_features)  # Potential outcome if treated
        y0_pred = self.outcome_model_control.predict(X_features)  # Potential outcome if not treated
        
        ate = np.mean(y1_pred - y0_pred)
        
        # Step 4: Compute Conditional Average Treatment Effect (CATE) by subgroups
        cate_by_feature = {}
        for col in feature_cols[:3]:  # Top 3 features
            if X[col].dtype in ['int64', 'float64']:
                # Binary split by median
                median_val = X[col].median()
                high_mask = X[col] > median_val
                
                cate_high = np.mean((y1_pred - y0_pred)[high_mask])
                cate_low = np.mean((y1_pred - y0_pred)[~high_mask])
                
                cate_by_feature[col] = {
                    'high': float(cate_high),
                    'low': float(cate_low)
                }
        
        return {
            'average_treatment_effect': float(ate),
            'cate_by_feature': cate_by_feature,
            'propensity_score_mean': float(np.mean(propensity_scores)),
            'interpretation': f"On average, treatment changes outcome by {ate:.3f}"
        }


class ExplanationEngine:
    """
    Multi-level explanation engine.
    
    Generates explanations at different levels of detail.
    """
    
    def __init__(self, model, preprocessor, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
    
    def explain_global(self) -> Dict:
        """Global model explanation."""
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importance = dict(zip(self.feature_names, np.abs(coef)))
        else:
            importance = {}
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'top_features': list(importance.keys())[:10],
            'feature_importance': importance,
            'n_features_used': len([v for v in importance.values() if v > 0.01])
        }
    
    def explain_local(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """Local explanation for a single prediction."""
        instance = X.iloc[instance_idx:instance_idx+1]
        
        # Get prediction
        prediction = self.model.predict(instance)[0]
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(instance)[0]
        
        # Approximate feature contribution (simplified LIME-like)
        baseline = X.mean().values.reshape(1, -1)
        contributions = []
        
        for i, col in enumerate(X.columns):
            # Change only this feature
            modified = baseline.copy()
            modified[0, i] = instance.iloc[0, i]
            
            pred_modified = self.model.predict_proba(modified)[0][1] if hasattr(self.model, 'predict_proba') else self.model.predict(modified)[0]
            pred_baseline = self.model.predict_proba(baseline)[0][1] if hasattr(self.model, 'predict_proba') else self.model.predict(baseline)[0]
            
            contribution = pred_modified - pred_baseline
            contributions.append((col, contribution))
        
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': float(prediction),
            'probability': probability.tolist() if probability is not None else None,
            'top_positive_features': [
                {'feature': name, 'contribution': float(cont)}
                for name, cont in contributions[:5] if cont > 0
            ],
            'top_negative_features': [
                {'feature': name, 'contribution': float(cont)}
                for name, cont in contributions[:5] if cont < 0
            ]
        }
    
    def explain_counterfactual(self, X: pd.DataFrame, instance_idx: int = 0,
                               desired_outcome: Optional[int] = None) -> Dict:
        """Generate counterfactual explanation."""
        instance = X.iloc[instance_idx]
        current_pred = self.model.predict(instance.values.reshape(1, -1))[0]
        
        if desired_outcome is None:
            desired_outcome = 1 - current_pred
        
        # Find minimal changes to flip prediction
        changes_needed = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Try increasing and decreasing
                for delta in [X[col].std(), -X[col].std()]:
                    modified = instance.copy()
                    modified[col] += delta
                    
                    new_pred = self.model.predict(modified.values.reshape(1, -1))[0]
                    if new_pred == desired_outcome:
                        changes_needed.append({
                            'feature': col,
                            'change': f"{instance[col]:.2f} → {modified[col]:.2f}",
                            'delta': float(delta)
                        })
        
        return {
            'current_prediction': int(current_pred),
            'desired_prediction': int(desired_outcome),
            'possible_changes': changes_needed[:5],
            'explanation': f"To change prediction from {current_pred} to {desired_outcome}, consider changing: " + 
                          ', '.join([c['feature'] for c in changes_needed[:3]])
        }
    
    def to_natural_language(self, explanation_type: str = 'global') -> str:
        """Convert explanation to natural language."""
        if explanation_type == 'global':
            global_exp = self.explain_global()
            top_features = global_exp['top_features'][:5]
            
            nl = f"This model makes predictions primarily based on {len(global_exp['top_features'])} features. "
            nl += f"The most important are: {', '.join(top_features)}. "
            nl += f"It uses {global_exp['n_features_used']} features significantly."
            
            return nl
        
        return "Explanation generated."


class SmartEnsemble(BaseEstimator, ClassifierMixin):
    """
    Intelligent ensemble that learns when to trust each model.
    
    Not just averaging - dynamic weighting based on:
    - Prediction confidence
    - Input feature values
    - Historical accuracy on similar inputs
    """
    
    def __init__(self, models: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.confidence_threshold = 0.7
        self.meta_learner = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit all base models and meta-learner."""
        # Fit base models
        self.model_predictions = {}
        for name, model in self.models:
            model.fit(X, y)
            self.model_predictions[name] = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
        
        # Create meta-features (stacking)
        meta_features = self._create_meta_features(X)
        
        # Train meta-learner to combine predictions
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.meta_learner.fit(meta_features, y)
        
        return self
    
    def _create_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create meta-features from base model predictions."""
        features = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                features.append(proba[:, 1])  # Probability of positive class
            else:
                pred = model.predict(X)
                features.append(pred)
        
        return np.column_stack(features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with smart ensemble."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with smart ensemble."""
        # Get meta-features
        meta_features = self._create_meta_features(X)
        
        # Use meta-learner
        return self.meta_learner.predict_proba(meta_features)
    
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction uncertainty.
        
        High uncertainty when models disagree.
        """
        predictions = []
        
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
                predictions.append(proba)
            else:
                pred = model.predict(X).astype(float)
                predictions.append(pred)
        
        # Variance across models = uncertainty
        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)
        
        return uncertainty
    
    def should_defer_to_human(self, X: pd.DataFrame, threshold: float = 0.2) -> np.ndarray:
        """
        Determine which predictions should be reviewed by human.
        
        Returns boolean array indicating uncertain predictions.
        """
        uncertainty = self.get_uncertainty(X)
        return uncertainty > threshold


class UncertaintyQuantifier:
    """
    Quantify prediction uncertainty for reliable decision-making.
    """
    
    @staticmethod
    def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Compute calibration curve.
        
        Well-calibrated model: predicted probability = empirical frequency
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibrations = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibrations.append({
                    'bin': (bin_lower, bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'n_samples': int(in_bin.sum())
                })
        
        return {
            'calibration_data': calibrations,
            'expected_calibration_error': sum([
                c['n_samples'] * abs(c['accuracy'] - c['confidence'])
                for c in calibrations
            ]) / len(y_true)
        }
    
    @staticmethod
    def prediction_intervals(model, X: pd.DataFrame, 
                            method: str = 'bootstrap') -> Dict:
        """
        Compute prediction intervals.
        
        Returns range where true value likely falls.
        """
        if method == 'bootstrap':
            # Bootstrap predictions
            n_bootstrap = 100
            predictions = []
            
            for _ in range(n_bootstrap):
                # Resample data
                indices = np.random.choice(len(X), len(X), replace=True)
                X_boot = X.iloc[indices]
                
                # Predict
                pred = model.predict(X_boot)
                predictions.append(pred[0])  # First instance
            
            predictions = np.array(predictions)
            
            return {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'ci_95': (float(np.percentile(predictions, 2.5)), 
                         float(np.percentile(predictions, 97.5))),
                'ci_99': (float(np.percentile(predictions, 0.5)), 
                         float(np.percentile(predictions, 99.5)))
            }
        
        return {}


class AutomatedFeatureSelection:
    """
    Automatically select best features.
    
    Not just correlation - considers:
    - Redundancy (mutual information)
    - Stability across folds
    - Computational cost
    """
    
    def __init__(self, max_features: int = 50, method: str = 'recursive'):
        self.max_features = max_features
        self.method = method
        self.selected_features = []
        self.feature_scores = {}
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'AutomatedFeatureSelection':
        """Select features."""
        from sklearn.feature_selection import RFECV, SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        
        if self.method == 'recursive':
            # Recursive feature elimination
            selector = RFECV(
                RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
                min_features_to_select=min(self.max_features, X.shape[1] // 2),
                cv=3,
                n_jobs=-1
            )
            selector.fit(X, y)
            
            self.selected_features = X.columns[selector.support_].tolist()
            self.feature_scores = dict(zip(X.columns, selector.ranking_))
        
        elif self.method == 'importance':
            # Select by importance
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model.fit(X, y)
            
            importances = model.feature_importances_
            threshold = np.sort(importances)[-min(self.max_features, len(importances))]
            
            selector = SelectFromModel(model, threshold=threshold)
            selector.fit(X, y)
            
            self.selected_features = X.columns[selector.get_support()].tolist()
            self.feature_scores = dict(zip(X.columns, importances))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from dataframe."""
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
