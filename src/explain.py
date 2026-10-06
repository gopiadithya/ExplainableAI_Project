import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from typing import Any, Dict, List, Optional, Tuple, Union

RANDOM_STATE = 42

def create_explainer(model: Any, X_train: Union[pd.DataFrame, np.ndarray]) -> Any:
    """
    Creates a SHAP TreeExplainer for tree-based models or KernelExplainer for linear/other models.
    """
    try:
        # Check if the model is tree-based by examining class name
        model_type = type(model).__name__.lower()
        if 'xgb' in model_type or 'randomforest' in model_type or 'gradientboosting' in model_type or 'lgbm' in model_type:
            return shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for non-tree models (e.g., LogisticRegression)
            # Use summary of background dataset for speed if it's too large
            background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
            return shap.KernelExplainer(model.predict_proba, background)
    except Exception as e:
        print(f"Failed to create explainer: {e}. Falling back to KernelExplainer.")
        background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
        return shap.KernelExplainer(model.predict_proba, background)

def compute_shap_values(explainer: Any, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> Any:
    """
    Computes SHAP values and returns them as a SHAP Explanation object.
    Handles the case where SHAP values might be 2D (binary classification) by returning values for class 1.
    """
    try:
        shap_values = explainer(X)
        
        # Ensure we have an Explanation object, fallback handling
        if not hasattr(shap_values, 'values'):
            # Convert raw numpy array to Explanation object if explainer(...) returned raw values
            shap_values = shap.Explanation(values=shap_values, data=X, feature_names=feature_names)
            
        # Handle binary classification for TreeExplainer where shape is (samples, features, classes)
        if len(shap_values.shape) > 2:
            # Always use values for class 1 (failure)
            shap_values = shap_values[:, :, 1]
            
        if feature_names is not None:
            shap_values.feature_names = feature_names
            
        return shap_values
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        raise

def plot_global_importance(shap_values: Any, feature_names: List[str], save_path: str = 'artifacts/figures/shap_global_importance.png') -> None:
    """
    Creates a SHAP bar plot showing mean absolute SHAP values for global feature importance.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.title('Global Feature Importance (Mean Absolute SHAP)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to create global importance plot: {e}")

def plot_beeswarm(shap_values: Any, save_path: str = 'artifacts/figures/shap_beeswarm.png') -> None:
    """
    Creates a SHAP beeswarm/summary plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.title('SHAP Summary Plot (Beeswarm)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to create beeswarm plot: {e}")

def plot_waterfall(shap_values: Any, index: int = 0, save_path: str = 'artifacts/figures/shap_waterfall.png') -> None:
    """
    Creates a SHAP waterfall plot for a single prediction.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure()
        shap.plots.waterfall(shap_values[index], show=False)
        plt.title(f'SHAP Waterfall Plot for Prediction {index}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to create waterfall plot: {e}")

def plot_dependence(shap_values: Any, feature_name: str, save_path: Optional[str] = None) -> None:
    """
    Creates a SHAP dependence plot for a specific feature.
    """
    try:
        if save_path is None:
            save_path = f"artifacts/figures/shap_dependence_{feature_name.replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure()
        shap.plots.scatter(shap_values[:, feature_name], color=shap_values, show=False)
        plt.title(f'SHAP Dependence Plot: {feature_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Failed to create dependence plot for {feature_name}: {e}")

def get_feature_contributions(shap_values: Any, feature_names: List[str], index: int = 0) -> List[Tuple[str, float, str]]:
    """
    Returns a sorted list of (feature_name, shap_value, direction) tuples for a single prediction.
    Direction is 'increases risk' or 'decreases risk'.
    """
    contributions = []
    sv = shap_values[index].values
    for i, val in enumerate(sv):
        direction = 'increases risk' if val > 0 else 'decreases risk'
        contributions.append((feature_names[i], float(val), direction))
    # Sort by absolute SHAP value (magnitude of contribution)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions

def generate_natural_language_explanation(prediction_proba: float, feature_contributions: List[Tuple[str, float, str]], threshold: float = 0.5) -> str:
    """
    Generates a deterministic natural language explanation based purely on SHAP values.
    """
    risk_level = "HIGH" if prediction_proba >= threshold else "LOW"
    
    # Separate top factors by direction (take top 3 max to avoid overwhelming)
    increases = []
    decreases = []
    
    for feat, val, direction in feature_contributions:
        # Avoid including 0-impact features
        if abs(val) < 1e-4:
            continue
        if direction == 'increases risk' and len(increases) < 3:
            increases.append(f"{feat} ({direction})")
        elif direction == 'decreases risk' and len(decreases) < 3:
            decreases.append(f"{feat} ({direction})")
            
    explanation = f"The model predicts a {risk_level} failure risk (probability: {prediction_proba:.2f})."
    
    if increases:
        explanation += f" The strongest factors contributing to this prediction are: {', '.join(increases)}."
    if decreases:
        explanation += f" Factors decreasing risk: {', '.join(decreases)}."
        
    return explanation

def get_risk_category(probability: float) -> str:
    """
    Returns risk category based on failure probability.
    """
    if probability < 0.3:
        return 'Low Risk'
    elif probability <= 0.6:
        return 'Medium Risk'
    else:
        return 'High Risk'

def save_explanation_report(shap_values: Any, feature_names: List[str], prediction_proba: float, save_path: str = 'artifacts/explanations/') -> None:
    """
    Saves a JSON report of the explanation.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        report = {
            "prediction_probability": float(prediction_proba),
            "risk_category": get_risk_category(prediction_proba),
            "feature_contributions": get_feature_contributions(shap_values, feature_names, index=0),
            "natural_language_explanation": generate_natural_language_explanation(
                prediction_proba, 
                get_feature_contributions(shap_values, feature_names, index=0)
            )
        }
        
        file_path = os.path.join(save_path, "explanation_report.json")
        with open(file_path, "w") as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        print(f"Failed to save explanation report: {e}")
