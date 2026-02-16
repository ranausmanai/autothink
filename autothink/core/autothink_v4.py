"""
AUTOTHINK V4 — Truly Intelligent One-Click AutoML

Auto-detects task type (binary, multiclass, regression).
Trains LightGBM + XGBoost + CatBoost ensemble with adaptive hyperparameters.
Optimizes blend weights via scipy on OOF predictions.
Intelligent verification and calibration.

Usage:
    from autothink import fit
    result = fit(df, target='Heart Disease')
    preds = result.predict(test_df)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
import time

from .preprocessing import IntelligentPreprocessor
from .feature_engineering_general import AdaptiveFeatureEngineer
from .validation import DataValidator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Detection
# ---------------------------------------------------------------------------

class TaskDetector:
    """Auto-detect task type, metric, and encode the target."""

    @staticmethod
    def detect(y: pd.Series) -> Dict:
        n_unique = y.nunique()
        dtype = y.dtype

        # Classification if: object/category, or numeric with few unique values
        if dtype == 'object' or dtype.name == 'category':
            task_type = 'binary' if n_unique == 2 else 'multiclass'
        elif n_unique <= 20:
            task_type = 'binary' if n_unique == 2 else 'multiclass'
        else:
            task_type = 'regression'

        # Encode target
        label_encoder = None
        if task_type in ('binary', 'multiclass'):
            if dtype == 'object' or dtype.name == 'category':
                label_encoder = LabelEncoder()
                encoded_y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            else:
                encoded_y = y.astype(int)
        else:
            encoded_y = y.astype(float)

        # Metric
        if task_type == 'binary':
            metric = 'auc'
        elif task_type == 'multiclass':
            metric = 'log_loss'
        else:
            metric = 'rmse'

        # Imbalance ratio (for classification)
        imbalance_ratio = 1.0
        if task_type in ('binary', 'multiclass'):
            counts = encoded_y.value_counts()
            imbalance_ratio = counts.max() / max(counts.min(), 1)

        return {
            'task_type': task_type,
            'metric': metric,
            'n_classes': n_unique if task_type != 'regression' else 0,
            'encoded_y': encoded_y,
            'label_encoder': label_encoder,
            'imbalance_ratio': imbalance_ratio,
        }


# ---------------------------------------------------------------------------
# Adaptive Hyperparameters
# ---------------------------------------------------------------------------

class AdaptiveHyperparams:
    """Generate model hyperparameters adapted to data characteristics."""

    @staticmethod
    def _base_tree_params(n_samples: int, n_features: int) -> Dict:
        """Shared heuristics for all tree-based models."""
        if n_samples < 2000:
            return dict(
                n_estimators=500, learning_rate=0.02,
                reg_alpha=1.0, reg_lambda=2.0,
                subsample=0.7, colsample=0.7,
                num_leaves=15, max_depth=5,
            )
        elif n_samples <= 50000:
            return dict(
                n_estimators=1500, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=1.0,
                subsample=0.8, colsample=0.8,
                num_leaves=31, max_depth=6,
            )
        else:
            return dict(
                n_estimators=3000, learning_rate=0.08,
                reg_alpha=0.1, reg_lambda=0.5,
                subsample=0.85, colsample=0.85,
                num_leaves=63, max_depth=7,
            )

    @classmethod
    def get_lgb_params(cls, n_samples: int, n_features: int,
                       task_type: str, imbalance_ratio: float) -> Dict:
        b = cls._base_tree_params(n_samples, n_features)
        objective = {
            'binary': 'binary',
            'multiclass': 'multiclass',
            'regression': 'regression',
        }[task_type]
        lgb_metric = {
            'binary': 'auc',
            'multiclass': 'multi_logloss',
            'regression': 'rmse',
        }[task_type]

        params = dict(
            objective=objective,
            metric=lgb_metric,
            n_estimators=b['n_estimators'],
            learning_rate=b['learning_rate'],
            num_leaves=b['num_leaves'],
            max_depth=b['max_depth'],
            reg_alpha=b['reg_alpha'],
            reg_lambda=b['reg_lambda'],
            feature_fraction=b['colsample'],
            bagging_fraction=b['subsample'],
            bagging_freq=5,
            verbose=-1,
            random_state=42,
        )
        if task_type == 'multiclass':
            params['num_class'] = None  # set externally
        if task_type == 'binary' and imbalance_ratio > 3:
            params['is_unbalance'] = True
        return params

    @classmethod
    def get_xgb_params(cls, n_samples: int, n_features: int,
                       task_type: str, imbalance_ratio: float) -> Dict:
        b = cls._base_tree_params(n_samples, n_features)
        objective = {
            'binary': 'binary:logistic',
            'multiclass': 'multi:softprob',
            'regression': 'reg:squarederror',
        }[task_type]
        eval_metric = {
            'binary': 'auc',
            'multiclass': 'mlogloss',
            'regression': 'rmse',
        }[task_type]

        params = dict(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=b['n_estimators'],
            learning_rate=b['learning_rate'],
            max_depth=b['max_depth'],
            reg_alpha=b['reg_alpha'],
            reg_lambda=b['reg_lambda'],
            subsample=b['subsample'],
            colsample_bytree=b['colsample'],
            tree_method='hist',
            random_state=42,
            verbosity=0,
        )
        if task_type == 'binary' and imbalance_ratio > 3:
            params['scale_pos_weight'] = imbalance_ratio
        return params

    @classmethod
    def get_cat_params(cls, n_samples: int, n_features: int,
                       task_type: str, imbalance_ratio: float) -> Dict:
        b = cls._base_tree_params(n_samples, n_features)
        loss = {
            'binary': 'Logloss',
            'multiclass': 'MultiClass',
            'regression': 'RMSE',
        }[task_type]

        params = dict(
            loss_function=loss,
            iterations=b['n_estimators'],
            learning_rate=b['learning_rate'],
            depth=min(b['max_depth'], 8),
            l2_leaf_reg=b['reg_lambda'],
            bootstrap_type='Bernoulli',
            subsample=b['subsample'],
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )
        if task_type == 'binary' and imbalance_ratio > 3:
            params['auto_class_weights'] = 'Balanced'
        return params


# ---------------------------------------------------------------------------
# Intelligent Ensemble
# ---------------------------------------------------------------------------

class IntelligentEnsemble:
    """Train LGB + XGB + CatBoost with K-fold, optimize blend weights."""

    def __init__(self):
        self.models = {'lgb': [], 'xgb': [], 'cat': []}
        self.blend_weights = None
        self.task_info = None
        self.n_folds = 5
        self.fold_scores = []  # per-fold score for each engine
        self.oof_preds = None
        self.calibrator = None

    # ---- scoring helpers ---------------------------------------------------

    @staticmethod
    def _score(y_true, y_pred, task_type: str, metric: str) -> float:
        if metric == 'auc':
            return roc_auc_score(y_true, y_pred)
        elif metric == 'log_loss':
            return -log_loss(y_true, y_pred)  # negate so higher=better
        else:  # rmse
            return -np.sqrt(mean_squared_error(y_true, y_pred))

    # ---- fold prediction helpers ------------------------------------------

    @staticmethod
    def _predict_fold_lgb(model, X, task_type):
        if task_type == 'regression':
            return model.predict(X)
        elif task_type == 'binary':
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)

    @staticmethod
    def _predict_fold_xgb(model, X, task_type):
        if task_type == 'regression':
            return model.predict(X)
        elif task_type == 'binary':
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)

    @staticmethod
    def _predict_fold_cat(model, X, task_type):
        if task_type == 'regression':
            return model.predict(X)
        elif task_type == 'binary':
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)

    # ---- fit --------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, task_info: Dict,
            verbose: bool = True) -> 'IntelligentEnsemble':
        self.task_info = task_info
        task_type = task_info['task_type']
        metric = task_info['metric']
        n_classes = task_info['n_classes']
        imbalance_ratio = task_info['imbalance_ratio']
        n_samples, n_features = X.shape

        self.n_folds = 3 if n_samples > 50000 else 5

        # --- adaptive hyperparams ---
        lgb_params = AdaptiveHyperparams.get_lgb_params(
            n_samples, n_features, task_type, imbalance_ratio)
        xgb_params = AdaptiveHyperparams.get_xgb_params(
            n_samples, n_features, task_type, imbalance_ratio)
        cat_params = AdaptiveHyperparams.get_cat_params(
            n_samples, n_features, task_type, imbalance_ratio)

        if task_type == 'multiclass':
            lgb_params['num_class'] = n_classes
            xgb_params['num_class'] = n_classes

        # --- prepare OOF containers ---
        if task_type in ('binary', 'regression'):
            oof_lgb = np.zeros(n_samples)
            oof_xgb = np.zeros(n_samples)
            oof_cat = np.zeros(n_samples)
        else:
            oof_lgb = np.zeros((n_samples, n_classes))
            oof_xgb = np.zeros((n_samples, n_classes))
            oof_cat = np.zeros((n_samples, n_classes))

        # --- fold splitter ---
        if task_type == 'regression':
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            split_iter = kf.split(X)
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                  random_state=42)
            split_iter = skf.split(X, y)

        fold_scores_lgb, fold_scores_xgb, fold_scores_cat = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(split_iter):
            if verbose:
                logger.info(f"   Fold {fold+1}/{self.n_folds}...")

            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # ----- LightGBM -----
            lgb_cls = lgb.LGBMRegressor if task_type == 'regression' else lgb.LGBMClassifier
            lgb_p = {k: v for k, v in lgb_params.items() if k != 'metric'}
            lgb_model = lgb_cls(**lgb_p)
            lgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            self.models['lgb'].append(lgb_model)
            oof_lgb[val_idx] = self._predict_fold_lgb(lgb_model, X_val, task_type)

            # ----- XGBoost -----
            xgb_cls = xgb.XGBRegressor if task_type == 'regression' else xgb.XGBClassifier
            xgb_p = {k: v for k, v in xgb_params.items() if k != 'eval_metric'}
            xgb_model = xgb_cls(**xgb_p)
            xgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models['xgb'].append(xgb_model)
            oof_xgb[val_idx] = self._predict_fold_xgb(xgb_model, X_val, task_type)

            # ----- CatBoost -----
            cat_cls = cb.CatBoostRegressor if task_type == 'regression' else cb.CatBoostClassifier
            cat_model = cat_cls(**cat_params)
            cat_model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
            )
            self.models['cat'].append(cat_model)
            oof_cat[val_idx] = self._predict_fold_cat(cat_model, X_val, task_type)

            # Score this fold
            if task_type in ('binary', 'regression'):
                s_lgb = self._score(y_val, oof_lgb[val_idx], task_type, metric)
                s_xgb = self._score(y_val, oof_xgb[val_idx], task_type, metric)
                s_cat = self._score(y_val, oof_cat[val_idx], task_type, metric)
            else:
                s_lgb = self._score(y_val, oof_lgb[val_idx], task_type, metric)
                s_xgb = self._score(y_val, oof_xgb[val_idx], task_type, metric)
                s_cat = self._score(y_val, oof_cat[val_idx], task_type, metric)

            fold_scores_lgb.append(s_lgb)
            fold_scores_xgb.append(s_xgb)
            fold_scores_cat.append(s_cat)

            if verbose:
                label = metric.upper()
                logger.info(f"LGB {label}={s_lgb:.5f}  XGB={s_xgb:.5f}  CAT={s_cat:.5f}")

        self.fold_scores = {
            'lgb': fold_scores_lgb,
            'xgb': fold_scores_xgb,
            'cat': fold_scores_cat,
        }

        # --- optimize blend weights ---
        self.blend_weights = self._optimize_blend(
            oof_lgb, oof_xgb, oof_cat, y, task_type, metric)

        w = self.blend_weights
        if verbose:
            logger.info(f"   Blend weights: LGB={w[0]:.3f}  XGB={w[1]:.3f}  CAT={w[2]:.3f}")

        # blended OOF
        self.oof_preds = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
        blend_score = self._score(y, self.oof_preds, task_type, metric)
        if verbose:
            logger.info(f"   Blended OOF {metric.upper()}: {blend_score:.5f}")

        # --- calibration for classification ---
        if task_type in ('binary', 'multiclass'):
            self._fit_calibrator(self.oof_preds, y, task_type, metric)

        return self

    # ---- blend optimization -----------------------------------------------

    def _optimize_blend(self, oof1, oof2, oof3, y,
                        task_type: str, metric: str) -> np.ndarray:
        """Find optimal blend weights via scipy minimize."""
        def objective(w):
            w = np.abs(w)
            w = w / w.sum()
            blended = w[0]*oof1 + w[1]*oof2 + w[2]*oof3
            return -self._score(y, blended, task_type, metric)

        try:
            result = minimize(objective, x0=[1/3, 1/3, 1/3],
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-4})
            w = np.abs(result.x)
            w = w / w.sum()
            return w
        except Exception:
            return np.array([1/3, 1/3, 1/3])

    # ---- calibration ------------------------------------------------------

    def _fit_calibrator(self, oof_preds, y, task_type, metric):
        """Platt scaling via logistic regression on OOF preds."""
        if task_type == 'binary':
            oof_2d = oof_preds.reshape(-1, 1)
        elif task_type == 'multiclass':
            oof_2d = oof_preds  # already (n, classes)
        else:
            return

        try:
            cal = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            cal.fit(oof_2d, y)
            cal_preds = cal.predict_proba(oof_2d)
            if task_type == 'binary':
                cal_preds_flat = cal_preds[:, 1]
            else:
                cal_preds_flat = cal_preds

            score_before = self._score(y, oof_preds, task_type, metric)
            score_after = self._score(y, cal_preds_flat, task_type, metric)

            if score_after > score_before:
                self.calibrator = cal
        except Exception:
            pass

    # ---- predict ----------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        task_type = self.task_info['task_type']
        w = self.blend_weights

        preds_all = []
        for key in ('lgb', 'xgb', 'cat'):
            fold_preds = []
            for model in self.models[key]:
                pred_fn = getattr(self, f'_predict_fold_{key}')
                fold_preds.append(pred_fn(model, X, task_type))
            # average across folds for this engine
            preds_all.append(np.mean(fold_preds, axis=0))

        # weighted blend
        blended = w[0]*preds_all[0] + w[1]*preds_all[1] + w[2]*preds_all[2]

        # calibration
        if self.calibrator is not None:
            if task_type == 'binary':
                blended = self.calibrator.predict_proba(
                    blended.reshape(-1, 1))[:, 1]
            elif task_type == 'multiclass':
                blended = self.calibrator.predict_proba(blended)

        return blended


# ---------------------------------------------------------------------------
# Result Verifier
# ---------------------------------------------------------------------------

class ResultVerifier:
    """Post-training diagnostics."""

    @staticmethod
    def verify(oof_preds, y, fold_scores: Dict, task_info: Dict,
               feature_importances=None, feature_names=None,
               verbose: bool = True) -> Dict:
        task_type = task_info['task_type']
        metric = task_info['metric']

        report = {'verdict': 'good', 'warnings': []}

        # 1. Fold stability
        all_scores = []
        for engine, scores in fold_scores.items():
            all_scores.extend(scores)
        score_std = np.std(all_scores)
        score_mean = np.mean(all_scores)

        if score_std > 0.05:
            report['warnings'].append(
                f"High fold variance (std={score_std:.4f}). Model may be unstable.")
            report['verdict'] = 'warning'

        # 2. Suspiciously perfect
        if task_type == 'binary' and score_mean > 0.995:
            report['warnings'].append(
                f"CV score ({score_mean:.5f}) suspiciously perfect — possible leakage.")
            report['verdict'] = 'problem'
        elif task_type == 'regression' and score_mean > -0.001:
            report['warnings'].append(
                "Near-zero RMSE — possible leakage or constant target.")
            report['verdict'] = 'problem'

        # 3. Per-engine mean scores (for overfitting check between engines)
        engine_means = {e: np.mean(s) for e, s in fold_scores.items()}

        # 4. Feature importances (top 10)
        top_features = []
        if feature_importances is not None and feature_names is not None:
            imp_idx = np.argsort(feature_importances)[::-1][:10]
            top_features = [(feature_names[i], feature_importances[i])
                            for i in imp_idx]

        report['score_mean'] = score_mean
        report['score_std'] = score_std
        report['engine_means'] = engine_means
        report['top_features'] = top_features

        if verbose:
            logger.info(f"{'='*60}")
            logger.info(f"Verification: {report['verdict'].upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"   Mean CV {metric.upper()}: {score_mean:.5f} (+/- {score_std:.5f})")
            for eng, m in engine_means.items():
                logger.info(f"   {eng.upper()} mean: {m:.5f}")
            if report['warnings']:
                for w in report['warnings']:
                    logger.warning(f"   {w}")
            if top_features:
                logger.info(f"   Top 10 features:")
                for name, imp in top_features:
                    logger.info(f"      {name:40s} {imp:.4f}")

        return report


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

def _quick_feature_selection(X: pd.DataFrame, y: pd.Series,
                             task_type: str, top_k: int = 150) -> list:
    """Keep top-K features by LightGBM importance. Fast single-tree fit."""
    if X.shape[1] <= top_k:
        return list(X.columns)

    if task_type == 'regression':
        m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                               num_leaves=31, verbose=-1, random_state=42)
    else:
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                num_leaves=31, verbose=-1, random_state=42)
    m.fit(X, y)
    importances = m.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    return [X.columns[i] for i in idx]


# ---------------------------------------------------------------------------
# AutoThink V4 — Main Orchestrator
# ---------------------------------------------------------------------------

class AutoThinkV4:
    """
    AutoThink V4 — Truly intelligent one-click AutoML.

    Supports binary classification, multiclass classification, and regression.
    Trains a LightGBM + XGBoost + CatBoost ensemble with adaptive hyper-
    parameters and optimized blend weights.
    """

    def __init__(self, time_budget: int = 600, verbose: bool = True):
        self.time_budget = time_budget
        self.verbose = verbose

        self.task_info = None
        self.preprocessor = None
        self.feature_engineer = None
        self.selected_features = None
        self.ensemble = None
        self.verification_report = None
        self.cv_score = None
        self.cv_std = None

    # ---- fit --------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str) -> 'AutoThinkV4':
        start = time.time()

        if self.verbose:
            logger.info("=" * 70)
            logger.info("AUTOTHINK V4 -- Truly Intelligent AutoML")
            logger.info("=" * 70)

        # ------------------------------------------------------------------
        # 1. Task detection
        # ------------------------------------------------------------------
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.task_info = TaskDetector.detect(y)
        y_enc = self.task_info['encoded_y']

        if self.verbose:
            tt = self.task_info['task_type']
            m = self.task_info['metric']
            logger.info(f"   Task: {tt}  |  Metric: {m}  |  "
                        f"Samples: {len(X):,}  |  Features: {X.shape[1]}")
            if self.task_info['label_encoder'] is not None:
                classes = list(self.task_info['label_encoder'].classes_)
                logger.info(f"   Classes: {classes}")

        # ------------------------------------------------------------------
        # 2. Drop ID columns
        # ------------------------------------------------------------------
        id_cols = [c for c in X.columns if 'id' in c.lower()]
        if id_cols:
            if self.verbose:
                logger.info(f"   Dropping ID columns: {id_cols}")
            X = X.drop(columns=id_cols)

        # ------------------------------------------------------------------
        # 3. Data validation
        # ------------------------------------------------------------------
        if self.verbose:
            logger.info("   Validating data...")
        validator = DataValidator()
        report = validator.validate_dataset(X, y_enc)
        if not report['is_valid']:
            for err in report['errors']:
                logger.warning(f"   [!] {err}")

        # ------------------------------------------------------------------
        # 4. Preprocessing
        # ------------------------------------------------------------------
        if self.verbose:
            logger.info("   Preprocessing...")
        self.preprocessor = IntelligentPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, y_enc)
        if self.verbose:
            logger.info(f"   -> {X_processed.shape[1]} features after preprocessing")

        # ------------------------------------------------------------------
        # 5. Feature engineering
        # ------------------------------------------------------------------
        if self.verbose:
            logger.info("   Feature engineering...")
        self.feature_engineer = AdaptiveFeatureEngineer()
        X_eng = self.feature_engineer.fit_transform(X_processed, y_enc)
        if self.verbose:
            logger.info(f"   -> {X_eng.shape[1]} features after engineering")

        # ------------------------------------------------------------------
        # 6. Feature selection
        # ------------------------------------------------------------------
        if self.verbose:
            logger.info("   Feature selection...")
        self.selected_features = _quick_feature_selection(
            X_eng, y_enc, self.task_info['task_type'], top_k=150)
        X_sel = X_eng[self.selected_features]
        if self.verbose:
            logger.info(f"   -> {len(self.selected_features)} features selected")

        # ------------------------------------------------------------------
        # 7. Train ensemble
        # ------------------------------------------------------------------
        if self.verbose:
            logger.info(f"   Training LGB + XGB + CatBoost ensemble "
                        f"({3 if len(X_sel) <= 50000 else 3}-fold)...")
        self.ensemble = IntelligentEnsemble()
        self.ensemble.fit(X_sel, y_enc, self.task_info, verbose=self.verbose)

        # ------------------------------------------------------------------
        # 8. Verification
        # ------------------------------------------------------------------
        feat_imp = None
        feat_names = None
        if self.ensemble.models['lgb']:
            lgb_model = self.ensemble.models['lgb'][0]
            feat_imp = lgb_model.feature_importances_
            feat_names = list(X_sel.columns)

        self.verification_report = ResultVerifier.verify(
            self.ensemble.oof_preds, y_enc,
            self.ensemble.fold_scores, self.task_info,
            feature_importances=feat_imp,
            feature_names=feat_names,
            verbose=self.verbose,
        )

        self.cv_score = self.verification_report['score_mean']
        self.cv_std = self.verification_report['score_std']

        elapsed = time.time() - start
        if self.verbose:
            logger.info(f"   Total time: {elapsed:.1f}s")

        return self

    # ---- predict ----------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Drop ID columns
        id_cols = [c for c in X.columns if 'id' in c.lower()]
        if id_cols:
            X = X.drop(columns=id_cols)

        X_processed = self.preprocessor.transform(X)
        X_eng = self.feature_engineer.transform(X_processed)

        # Align to selected features (handle missing gracefully)
        for col in self.selected_features:
            if col not in X_eng.columns:
                X_eng[col] = 0
        X_sel = X_eng[self.selected_features]

        return self.ensemble.predict(X_sel)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fit_v4(df: pd.DataFrame, target: str, **kwargs) -> AutoThinkV4:
    """One-line AutoThink V4.

    Example:
        >>> from autothink import fit_v4
        >>> result = fit_v4(df, target='Heart Disease')
        >>> predictions = result.predict(test_df)
    """
    engine = AutoThinkV4(**kwargs)
    return engine.fit(df, target)
