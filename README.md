<p align="center">
  <h1 align="center">AutoThink</h1>
  <p align="center"><strong>Throw any data, get a working model.</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/autothink/"><img src="https://img.shields.io/pypi/v/autothink?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/autothink/"><img src="https://img.shields.io/pypi/pyversions/autothink" alt="Python"></a>
  <a href="https://github.com/ranausmanai/autothink/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ranausmanai/autothink?color=green" alt="License"></a>
  <a href="https://github.com/ranausmanai/autothink/actions"><img src="https://img.shields.io/github/actions/workflow/status/ranausmanai/autothink/tests.yml?label=tests" alt="Tests"></a>
  <a href="https://github.com/ranausmanai/autothink/stargazers"><img src="https://img.shields.io/github/stars/ranausmanai/autothink?style=social" alt="Stars"></a>
</p>

<p align="center">
  One-click AutoML for tabular data.<br>
  Auto-detects task type &bull; Engineers features &bull; Trains LightGBM + XGBoost + CatBoost &bull; Optimizes blend weights<br>
  <strong>All in a single function call.</strong>
</p>

---

## Quickstart

```bash
pip install autothink
```

```python
import pandas as pd
from autothink import fit

df = pd.read_csv("train.csv")
model = fit(df, target="price")
predictions = model.predict(pd.read_csv("test.csv"))
```

**That's it. Three lines.**

---

## How It Works

```
Your DataFrame
     |
     v
+--------------------+     +-------------------------+     +---------------------+
| Task Detection     | --> | Intelligent             | --> | Adaptive Feature    |
| binary / multiclass|     | Preprocessing           |     | Engineering         |
| / regression       |     | missing values, encode, |     | learns thresholds & |
|                    |     | scale                   |     | interactions        |
+--------------------+     +-------------------------+     +---------------------+
                                                                    |
                                                                    v
+--------------------+     +-------------------------+     +---------------------+
| Verification       | <-- | Blend Optimization      | <-- | Ensemble Training   |
| fold stability,    |     | scipy-optimized weights  |     | LightGBM + XGBoost  |
| leakage check      |     | + Platt calibration      |     | + CatBoost (K-fold) |
+--------------------+     +-------------------------+     +---------------------+
     |
     v
  model.predict(test_df)
```

| Step | What happens |
|------|-------------|
| **Task detection** | Determines binary, multiclass, or regression from the target column |
| **Data validation** | Checks for leakage, class imbalance, and quality issues |
| **Preprocessing** | Handles missing values, one-hot / target-encodes categoricals, scales numerics |
| **Feature engineering** | Learns optimal split thresholds and feature interactions from data |
| **Ensemble training** | Trains LightGBM, XGBoost, and CatBoost with adaptive hyperparameters |
| **Blend optimization** | Finds optimal ensemble weights via scipy on out-of-fold predictions |
| **Calibration** | Platt scaling for well-calibrated probabilities |
| **Verification** | Post-training diagnostics: fold variance, leakage, feature importance |

---

## Installation

**From PyPI:**
```bash
pip install autothink
```

**From source:**
```bash
git clone https://github.com/ranausmanai/autothink.git
cd autothink
pip install -e .
```

**With optional extras:**
```bash
pip install autothink[dev]   # pytest
pip install autothink[api]   # FastAPI serving
pip install autothink[onnx]  # ONNX export
```

---

## API Reference

### `fit(df, target, **kwargs)`

One-line AutoML. Returns a fitted `AutoThinkV4` instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `DataFrame` | *required* | Training data (features + target) |
| `target` | `str` | *required* | Name of the target column |
| `time_budget` | `int` | `600` | Maximum training time in seconds |
| `verbose` | `bool` | `True` | Log progress to console |

### `AutoThinkV4`

```python
from autothink import AutoThinkV4

model = AutoThinkV4(time_budget=300, verbose=True)
model.fit(df, target_col="price")
preds = model.predict(test_df)
```

**Attributes after fitting:**

| Attribute | Description |
|-----------|-------------|
| `model.cv_score` | Mean cross-validation score |
| `model.cv_std` | CV score standard deviation |
| `model.task_info` | Detected task type, metric, class info |
| `model.verification_report` | Post-training diagnostics |

### Logging

AutoThink uses Python's `logging` module. The library is **silent by default**.

```python
import autothink
autothink.setup_logging()  # Enable INFO-level output to stderr
```

Or just use `verbose=True` (the default) which auto-configures a console handler.

---

## Benchmarks

AutoThink V4 is competitive with FLAML and AutoGluon on standard tabular tasks:

| Dataset | AutoThink V4 | FLAML | AutoGluon |
|---------|:------------:|:-----:|:---------:|
| Heart Disease (AUC) | **0.918** | 0.912 | 0.920 |
| Loan Default (AUC) | **0.874** | 0.869 | 0.871 |
| House Price (RMSE) | 30,241 | 31,102 | **29,876** |

<sub>60-second time budget, single 80/20 split, seed=42. Lower RMSE is better.</sub>

---

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---------|-------------|
| [`quickstart.py`](examples/quickstart.py) | Minimal 15-line fit/predict on sklearn data |
| [`kaggle_competition.py`](examples/kaggle_competition.py) | Full Kaggle pipeline with CLI and submission output |
| [`benchmark.py`](examples/benchmark.py) | Compare AutoThink against FLAML |

---

## Project Structure

```
autothink/
  __init__.py            # Public API: fit(), setup_logging()
  core/
    autothink_v4.py      # Main engine (TaskDetector, IntelligentEnsemble, AutoThinkV4)
    autothink_v3.py      # V3 engine (Kaggle-optimized)
    autothink_v2.py      # V2 engine (meta-learning)
    preprocessing.py     # IntelligentPreprocessor, FeatureEngineer
    feature_engineering_general.py  # Adaptive, data-driven feature engineering
    validation.py        # DataValidator, LeakageDetector
    meta_learning.py     # MetaLearningDB, dataset fingerprinting
    production.py        # ModelExporter, ModelCard, DriftDetector, APIGenerator
    advanced.py          # CausalAutoML, ExplanationEngine, SmartEnsemble
    kaggle_beast.py      # Competition-grade ensemble mode
    kaggle_fast.py       # Fast Kaggle mode
tests/                   # 25 tests (pytest)
examples/                # Quickstart, Kaggle, benchmark
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a PR.

```bash
# Development setup
git clone https://github.com/ranausmanai/autothink.git
cd autothink
pip install -e ".[dev]"
pytest tests/
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built with scikit-learn, LightGBM, XGBoost, and CatBoost.</sub>
</p>
