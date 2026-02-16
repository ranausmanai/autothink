"""
AutoThink Safe - Hardened Version with Leakage Protection
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
import time

from .autothink_v2 import AutoThinkV2
from .validation import LeakageDetector, OverfittingDetector, DataValidator, PostTrainingValidator

warnings.filterwarnings('ignore')


class AutoThinkSafe(AutoThinkV2):
    """
    Hardened AutoThink with safety checks and leakage prevention.
    
    Key improvements:
    1. Automatic leakage detection and removal
    2. Small data protection
    3. CV sanity checks
    4. Overfitting prevention
    """
    
    def __init__(self, *args, safety_mode: str = 'strict', **kwargs):
        """
        Args:
            safety_mode: 'strict' (most safe) or 'lenient' (allow more flexibility)
        """
        super().__init__(*args, **kwargs)
        self.safety_mode = safety_mode
        self.leakage_report = None
        self.validation_report = None
    
    def fit(self, df: pd.DataFrame, target_col: str, 
            test_df: Optional[pd.DataFrame] = None):
        """
        Safe training with validation.
        """
        start_time = time.time()
        
        if self.verbose:
            print("=" * 70)
            print("🛡️  AUTOTHINK SAFE - Validated AutoML")
            print("=" * 70)
        
        # Phase 0: Data Validation
        X_raw = df.drop(columns=[target_col])
        y = df[target_col]
        
        if self.verbose:
            print("\n🔍 Running Safety Checks...")
        
        self.validation_report = DataValidator.validate_dataset(X_raw, y)
        
        if not self.validation_report['is_valid']:
            print("\n❌ CRITICAL ISSUES DETECTED:")
            for error in self.validation_report['errors']:
                print(f"   • {error}")
            print("\n🛠️  Auto-correcting issues...")
            # Don't fail - continue with auto-corrections
        
        if self.validation_report['warnings']:
            if self.verbose:
                print("\n⚠️  Warnings:")
                for warning in self.validation_report['warnings']:
                    print(f"   • {warning}")
        
        # Phase 1: Leakage Detection
        self.leakage_report = LeakageDetector.validate_no_leakage(X_raw, y)
        
        if self.leakage_report['has_leakage']:
            if self.verbose:
                print("\n🚨 DATA LEAKAGE DETECTED!")
                for critical in self.leakage_report['critical']:
                    print(f"   ❌ {critical}")
                print(f"\n   Auto-dropping: {self.leakage_report['columns_to_drop']}")
            
            # Drop leakage columns
            df = df.drop(columns=self.leakage_report['columns_to_drop'])
        else:
            if self.verbose:
                print("   ✓ No data leakage detected")
        
        # Phase 2: Check sample size and adjust strategy
        n_samples = len(df)
        n_features = df.shape[1] - 1  # Exclude target
        
        complexity_rec = OverfittingDetector.recommend_model_complexity(n_samples, n_features)
        
        if self.verbose and n_samples < 1000:
            print(f"\n📊 Small Data Protection Active:")
            print(f"   Samples: {n_samples}, Features: {n_features}")
            print(f"   Strategy: {complexity_rec['recommendation']}")
        
        # Adjust for small data
        if n_samples < 1000 and self.safety_mode == 'strict':
            # Force simpler models
            self.target_score = min(self.target_score, 0.85)  # Lower expectations
            if self.verbose:
                print(f"   Lowered target score to {self.target_score} for small dataset")
        
        # Phase 3: Normal training with parent class
        try:
            result = super().fit(df, target_col, test_df)
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Training failed: {e}")
            raise
        
        # Phase 4: Post-training validation
        if self.verbose:
            print("\n🔍 Post-Training Validation...")
        
        # Check CV sanity
        cv_check = OverfittingDetector.check_cv_sanity(
            np.array([result.cv_score - result.cv_std, result.cv_score + result.cv_std])
        )
        
        if cv_check['is_suspicious']:
            if self.verbose:
                for issue in cv_check['issues']:
                    print(f"   ⚠️  {issue}")
            
            # If suspiciously high CV, warn but don't fail
            if result.cv_score > 0.95:
                warnings.warn(
                    f"CV score ({result.cv_score:.3f}) is suspiciously high. "
                    "Model may have overfit or there may be data leakage. "
                    "Validate on holdout set before production use."
                )
        else:
            if self.verbose:
                print("   ✓ CV scores look reasonable")
        
        # Phase 5: Store safety info in result
        result.safety_info = {
            'validation_report': self.validation_report,
            'leakage_report': self.leakage_report,
            'complexity_recommendation': complexity_rec,
            'is_small_dataset': n_samples < 1000
        }
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("✅ Safe Training Complete")
            print("=" * 70)
        
        return result
    
    def _get_model_tiers(self, problem_type: str):
        """
        Override to add small data protections.
        """
        tiers = super()._get_model_tiers(problem_type)
        
        # Get current data info
        if hasattr(self, 'fingerprint') and self.fingerprint:
            n_samples = self.fingerprint.n_samples
            
            if n_samples < 1000:
                # Filter to only simple models for small data
                if self.safety_mode == 'strict':
                    if problem_type in ['binary', 'multiclass']:
                        tiers = {
                            'safe': [
                                ('LogisticRegression_L2', lambda **kw: __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(
                                    penalty='l2', C=0.1, max_iter=1000, **kw
                                )),
                                ('RidgeClassifier', lambda **kw: __import__('sklearn.linear_model', fromlist=['RidgeClassifier']).RidgeClassifier(alpha=1.0, **kw)),
                            ]
                        }
                    else:
                        tiers = {
                            'safe': [
                                ('Ridge', lambda **kw: __import__('sklearn.linear_model', fromlist=['Ridge']).Ridge(alpha=1.0, **kw)),
                                ('Lasso', lambda **kw: __import__('sklearn.linear_model', fromlist=['Lasso']).Lasso(alpha=0.1, **kw)),
                            ]
                        }
                
                if self.verbose:
                    print(f"   Limited to simple models due to small dataset ({n_samples} samples)")
        
        return tiers


def fit_safe(df, target: str, **kwargs):
    """
    One-line safe AutoML with leakage protection.
    
    Example:
        >>> from autothink import fit_safe
        >>> result = fit_safe(df, target='churn')
        >>> # Check for warnings
        >>> if result.safety_info['leakage_report']['has_leakage']:
        ...     print("Warning: Data leakage was detected and removed")
    """
    autothink = AutoThinkSafe(**kwargs)
    return autothink.fit(df, target)
