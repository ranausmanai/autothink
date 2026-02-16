"""
Production Features - Phase 4
Export, deployment, monitoring
"""

import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings
import pandas as pd
import numpy as np

try:
    import onnx
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelExporter:
    """
    Export models to various formats for production.
    """
    
    @staticmethod
    def to_pickle(model, filepath: str) -> str:
        """Export to pickle format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        return str(filepath)
    
    @staticmethod
    def to_joblib(model, filepath: str) -> str:
        """Export to joblib format (better for large models)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, filepath)
        return str(filepath)
    
    @staticmethod
    def to_onnx(model, feature_names: list, filepath: str) -> Optional[str]:
        """Export to ONNX format for cross-platform inference."""
        if not ONNX_AVAILABLE:
            warnings.warn("ONNX not installed. Install with: pip install skl2onnx onnx")
            return None
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
            
            # Convert
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save
            with open(filepath, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            return str(filepath)
        except Exception as e:
            warnings.warn(f"Could not convert to ONNX: {e}")
            return None


class ModelCard:
    """
    Generate model cards for documentation and transparency.
    
    Based on Google's Model Cards paper:
    https://modelcards.withgoogle.com/about
    """
    
    def __init__(self, result, dataset_name: str = "unknown"):
        self.result = result
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().isoformat()
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete model card."""
        return {
            "model_details": self._model_details(),
            "intended_use": self._intended_use(),
            "factors": self._factors(),
            "metrics": self._metrics(),
            "evaluation_data": self._evaluation_data(),
            "training_data": self._training_data(),
            "quantitative_analyses": self._quantitative_analyses(),
            "ethical_considerations": self._ethical_considerations(),
            "caveats": self._caveats(),
            "generated_at": self.timestamp
        }
    
    def _model_details(self) -> Dict:
        """Model architecture details."""
        return {
            "name": f"AutoThink_{self.result.model_name}",
            "version": "2.0",
            "type": self.result.model_name,
            "description": f"Automatically trained {self.result.model_name} model",
            "preprocessing": {
                "encoder": "IntelligentPreprocessor",
                "feature_engineering": self.result.feature_engineer is not None
            }
        }
    
    def _intended_use(self) -> Dict:
        """Intended and unintended uses."""
        return {
            "primary_use_cases": [
                "Automated prediction on similar tabular data"
            ],
            "users": "Data scientists, ML engineers",
            "out_of_scope": [
                "Data outside training distribution",
                "Adversarial inputs",
                "High-stakes decisions without human review"
            ]
        }
    
    def _factors(self) -> Dict:
        """Relevant factors for evaluation."""
        fp = self.result.fingerprint
        return {
            "dataset_size": fp.n_samples,
            "n_features": fp.n_features,
            "feature_types": {
                "numerical": fp.n_numerical,
                "categorical": fp.n_categorical
            },
            "class_balance": f"{fp.target_imbalance_ratio:.1f}:1"
        }
    
    def _metrics(self) -> Dict:
        """Performance metrics."""
        return {
            "cv_score": float(self.result.cv_score),
            "cv_std": float(self.result.cv_std),
            "test_score": float(self.result.test_score) if self.result.test_score else None,
            "training_time_seconds": float(self.result.train_time)
        }
    
    def _evaluation_data(self) -> Dict:
        """Evaluation dataset details."""
        return {
            "type": "5-fold cross-validation",
            "size": self.result.fingerprint.n_samples,
            "split": "Stratified" if self.result.fingerprint.target_cardinality == 2 else "Random"
        }
    
    def _training_data(self) -> Dict:
        """Training data details."""
        return {
            "source": self.dataset_name,
            "size": self.result.fingerprint.n_samples,
            "missing_values": f"{self.result.fingerprint.missing_ratio:.1%}"
        }
    
    def _quantitative_analyses(self) -> Dict:
        """Feature importance."""
        importance = self.result.feature_importance
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            return {
                "top_features": [
                    {"name": name, "importance": float(imp)}
                    for name, imp in top_features
                ]
            }
        return {}
    
    def _ethical_considerations(self) -> Dict:
        """Ethical considerations."""
        return {
            "risks": [
                "Model may perpetuate biases in training data",
                "Not suitable for high-stakes decisions without human oversight"
            ],
            "mitigations": [
                "Cross-validation used for robust evaluation",
                "Feature importance provided for transparency"
            ]
        }
    
    def _caveats(self) -> Dict:
        """Known limitations."""
        return {
            "general": [
                "Performance may degrade on data outside training distribution",
                "Feature engineering is automated and may miss domain-specific insights"
            ],
            "recommendations": [
                "Monitor model performance in production",
                "Retrain when data distribution changes",
                "Validate predictions on holdout set regularly"
            ]
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        card = self.generate()
        
        md = f"""# Model Card: {card['model_details']['name']}

## Model Details
- **Name**: {card['model_details']['name']}
- **Type**: {card['model_details']['type']}
- **Generated**: {card['generated_at']}

## Intended Use
**Primary Use Cases**: {', '.join(card['intended_use']['primary_use_cases'])}

**Out of Scope**: {', '.join(card['intended_use']['out_of_scope'])}

## Performance Metrics
- **CV Score**: {card['metrics']['cv_score']:.4f} (+/- {card['metrics']['cv_std']:.4f})
"""
        if card['metrics']['test_score']:
            md += f"- **Test Score**: {card['metrics']['test_score']:.4f}\n"
        
        md += f"""
## Training Data
- **Samples**: {card['training_data']['size']}
- **Features**: {self.result.fingerprint.n_features}
- **Missing Values**: {card['training_data']['missing_values']}

## Top Features
"""
        if 'top_features' in card['quantitative_analyses']:
            for feat in card['quantitative_analyses']['top_features']:
                md += f"- {feat['name']}: {feat['importance']:.4f}\n"
        
        md += """
## Ethical Considerations
**Risks**: Model may perpetuate biases present in training data.

**Recommendation**: Regular bias audits and human oversight for high-stakes decisions.

## Caveats
"""
        for caveat in card['caveats']['general']:
            md += f"- {caveat}\n"
        
        return md
    
    def save(self, filepath: str):
        """Save model card to JSON and Markdown."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.generate(), f, indent=2)
        
        # Markdown
        with open(filepath.with_suffix('.md'), 'w') as f:
            f.write(self.to_markdown())


class DriftDetector:
    """
    Detect data drift in production.
    
    Types of drift:
    - Data drift: Input distribution changes
    - Concept drift: Relationship X→y changes
    - Label drift: Target distribution changes
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize with reference (training) data.
        
        Args:
            reference_data: Training data distribution
        """
        self.reference = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute distribution statistics."""
        stats = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'q05': df[col].quantile(0.05),
                'q95': df[col].quantile(0.95),
                'missing': df[col].isnull().mean()
            }
        
        for col in df.select_dtypes(include=['object']).columns:
            stats[col] = {
                'unique': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict(),
                'missing': df[col].isnull().mean()
            }
        
        return stats
    
    def detect(self, new_data: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """
        Detect drift in new data.
        
        Returns:
            Dictionary with drift status and details
        """
        alerts = []
        
        # Check for new columns
        new_cols = set(new_data.columns) - set(self.reference.columns)
        if new_cols:
            alerts.append({
                'type': 'schema_drift',
                'severity': 'high',
                'message': f'New columns detected: {new_cols}'
            })
        
        # Check for missing columns
        missing_cols = set(self.reference.columns) - set(new_data.columns)
        if missing_cols:
            alerts.append({
                'type': 'schema_drift',
                'severity': 'high',
                'message': f'Missing columns: {missing_cols}'
            })
        
        # Check numerical features for distribution shift
        for col in self.reference.select_dtypes(include=[np.number]).columns:
            if col not in new_data.columns:
                continue
            
            ref_stats = self.reference_stats[col]
            new_mean = new_data[col].mean()
            new_std = new_data[col].std()
            
            # Z-score test for mean shift
            if ref_stats['std'] > 0:
                z_score = abs(new_mean - ref_stats['mean']) / ref_stats['std']
                if z_score > 3:  # 3 sigma rule
                    alerts.append({
                        'type': 'data_drift',
                        'feature': col,
                        'severity': 'medium',
                        'message': f'Mean shift in {col}: {ref_stats["mean"]:.2f} → {new_mean:.2f}'
                    })
        
        # Check categorical features
        for col in self.reference.select_dtypes(include=['object']).columns:
            if col not in new_data.columns:
                continue
            
            ref_unique = self.reference[col].nunique()
            new_unique = new_data[col].nunique()
            
            if new_unique > ref_unique * 1.5:
                alerts.append({
                    'type': 'data_drift',
                    'feature': col,
                    'severity': 'low',
                    'message': f'Cardinality increased in {col}: {ref_unique} → {new_unique}'
                })
        
        return {
            'drift_detected': len(alerts) > 0,
            'n_alerts': len(alerts),
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }


class APIGenerator:
    """
    Generate FastAPI code for model serving.
    """
    
    @staticmethod
    def generate_fastapi_code(model_path: str, feature_names: list, output_path: str):
        """Generate FastAPI application code."""
        
        code = f'''"""
Auto-generated FastAPI application for model serving.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
from pathlib import Path

# Load model
model_path = Path("{model_path}")
model = joblib.load(model_path)

app = FastAPI(
    title="AutoThink Model API",
    description="Automatically generated ML model API",
    version="1.0"
)

class PredictionRequest(BaseModel):
    """Input data schema."""
{chr(10).join([f'    {name}: float = Field(..., description="Feature {name}")' for name in feature_names[:5]])}
    # ... (add remaining features)

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: float
    probability: float = None
    confidence: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make prediction."""
    try:
        # Convert to array
        features = np.array([[{', '.join([f'request.{name}' for name in feature_names[:5]])}]])
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(features)[0][1])
        
        # Determine confidence
        confidence = "high" if probability and (probability > 0.8 or probability < 0.2) else "medium"
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: List[Dict]):
    """Batch prediction endpoint."""
    try:
        features = np.array([[row.get(name, 0) for name in {feature_names}] for row in data])
        predictions = model.predict(features).tolist()
        return {{"predictions": predictions, "count": len(predictions)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        return output_path
