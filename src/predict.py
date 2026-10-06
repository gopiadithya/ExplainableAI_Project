"""
Prediction module for the X-Maintain project.
Handles loading production model and making predictions with explanations.
"""
import os
import json
import copy
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional

from src.explain import (
    compute_shap_values,
    get_feature_contributions,
    generate_natural_language_explanation,
    get_risk_category
)

RANDOM_STATE = 42


def load_production_model(model_dir: str = 'models/') -> Tuple[Any, Any, List[str]]:
    """
    Loads the best model, preprocessor, and feature names from saved files.

    Returns:
        Tuple of (model, preprocessor, feature_names)
    """
    # Try .joblib first, then .pkl for backward compatibility
    model_path = os.path.join(model_dir, 'best_model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'model.pkl')

    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    if not os.path.exists(preprocessor_path):
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')

    feature_names_path = os.path.join(model_dir, 'feature_names.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_dir}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found in {model_dir}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found in {model_dir}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)

    return model, preprocessor, feature_names


def predict_single(model: Any, preprocessor: Any, feature_names: List[str],
                   input_data: Union[Dict, pd.DataFrame]) -> Tuple[int, float]:
    """
    Predicts class and failure probability for a single instance.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessing pipeline
        feature_names: List of feature names after preprocessing
        input_data: Raw input data (dict or DataFrame with original feature columns)

    Returns:
        Tuple of (predicted_class, failure_probability)
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Transform input through preprocessor
    processed_input = preprocessor.transform(df_input)

    # Predict
    predicted_class = int(model.predict(processed_input)[0])

    # Get probability
    if hasattr(model, 'predict_proba'):
        failure_probability = float(model.predict_proba(processed_input)[0][1])
    else:
        failure_probability = 1.0 if predicted_class == 1 else 0.0

    return predicted_class, failure_probability


def predict_with_explanation(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    explainer: Any,
    input_data: Union[Dict, pd.DataFrame]
) -> Tuple[int, float, str, List[Tuple[str, float, str]], str]:
    """
    Returns prediction details along with SHAP explanations and natural language summary.

    Returns:
        Tuple of (predicted_class, failure_probability, risk_category,
                  feature_contributions, natural_language_explanation)
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Get prediction
    predicted_class, failure_probability = predict_single(
        model, preprocessor, feature_names, input_data
    )
    risk_category = get_risk_category(failure_probability)

    # Compute SHAP explanations on processed input
    processed_input = preprocessor.transform(df_input)
    processed_df = pd.DataFrame(processed_input, columns=feature_names)

    shap_values = compute_shap_values(explainer, processed_df, feature_names)
    feature_contributions = get_feature_contributions(shap_values, feature_names, index=0)
    nl_explanation = generate_natural_language_explanation(
        failure_probability, feature_contributions
    )

    return predicted_class, failure_probability, risk_category, feature_contributions, nl_explanation


def what_if_analysis(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    explainer: Any,
    base_input: Dict[str, Any],
    modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs model-based what-if analysis by comparing predictions for
    original vs modified inputs.

    IMPORTANT: This is purely model-based sensitivity analysis, NOT causal inference.
    Changing a feature value in the model does not guarantee the same outcome
    in a real industrial setting.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessing pipeline
        feature_names: Feature names after preprocessing
        explainer: SHAP explainer
        base_input: Original input values
        modifications: Dict of feature_name -> new_value to modify

    Returns:
        Dictionary with original and modified predictions and analysis metadata.
    """
    # 1. Evaluate base input
    base_class, base_proba = predict_single(
        model, preprocessor, feature_names, base_input
    )

    # 2. Create modified input
    modified_input = copy.deepcopy(base_input)
    changes = []

    for feature, new_value in modifications.items():
        if feature in modified_input:
            old_value = modified_input[feature]
            modified_input[feature] = new_value
            changes.append({
                "feature": feature,
                "old_value": old_value,
                "new_value": new_value
            })

    # 3. Predict modified input
    mod_class, mod_proba = predict_single(
        model, preprocessor, feature_names, modified_input
    )

    # 4. Return results
    return {
        "analysis_type": "Model-based What-If Analysis (Not Causal)",
        "disclaimer": (
            "This analysis shows how the model's prediction changes when input "
            "features are modified. It does NOT imply that physically changing "
            "these parameters will guarantee the predicted outcome."
        ),
        "original_prediction": base_class,
        "modified_prediction": mod_class,
        "original_proba": base_proba,
        "modified_proba": mod_proba,
        "proba_difference": mod_proba - base_proba,
        "original_risk": get_risk_category(base_proba),
        "modified_risk": get_risk_category(mod_proba),
        "changes": changes
    }
