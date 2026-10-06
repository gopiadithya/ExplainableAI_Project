"""
Training module for the X-Maintain project.
Trains Logistic Regression, Random Forest, and XGBoost models.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_raw_data
from src.preprocessing import (
    RANDOM_STATE, NUMERICAL_COLS, TARGET_COL,
    remove_leakage_columns, encode_categorical,
    get_feature_target_split, create_train_test_split,
    build_preprocessing_pipeline,
    sanitize_feature_names, get_sanitized_feature_names
)


def get_models(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """Returns a dictionary of configured models.

    Args:
        scale_pos_weight: Weight for the positive class in XGBoost.
            Should be computed as (num_negative / num_positive).

    Returns:
        Dictionary mapping model names to configured model instances.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fits and returns the model."""
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Trains all models and returns a dictionary of name -> trained model."""
    num_negative = int((y_train == 0).sum())
    num_positive = int((y_train == 1).sum())
    scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    print(f"  Class distribution: {num_negative} negative, {num_positive} positive")
    print(f"  scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    models = get_models(scale_pos_weight=scale_pos_weight)
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        trained_models[name] = train_model(model, X_train, y_train)
        print(f"  {name} trained successfully.")
    return trained_models


def save_model(model: Any, name: str, output_dir: str = 'models/') -> str:
    """Saves a model using joblib. Returns the filepath."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.joblib")
    joblib.dump(model, filepath)
    return filepath


def load_model(name: str, model_dir: str = 'models/') -> Any:
    """Loads a model using joblib."""
    filepath = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)


if __name__ == '__main__':
    print("=" * 60)
    print("X-Maintain: Model Training Pipeline")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/7] Loading raw data...")
    df = load_raw_data()
    print(f"  Dataset shape: {df.shape}")

    # 2. Remove leakage columns
    print("\n[2/7] Removing leakage and ID columns...")
    df_clean, removed_info = remove_leakage_columns(df)
    for col, reason in removed_info.items():
        print(f"  Removed '{col}': {reason}")

    # 3. Encode categoricals
    print("\n[3/7] Encoding categorical columns...")
    df_encoded = encode_categorical(df_clean)
    print(f"  Shape after encoding: {df_encoded.shape}")

    # 4. Split features/target
    print("\n[4/7] Splitting features and target...")
    X, y = get_feature_target_split(df_encoded)
    print(f"  Features: {X.shape}, Target: {y.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    # 5. Create train/val/test split
    print("\n[5/7] Creating stratified train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 6. Build preprocessing pipeline and fit on train data
    print("\n[6/7] Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()

    feature_names = get_sanitized_feature_names()

    X_train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=feature_names, index=X_train.index
    )
    X_val_processed = pd.DataFrame(
        preprocessor.transform(X_val),
        columns=feature_names, index=X_val.index
    )
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=feature_names, index=X_test.index
    )
    print(f"  Processed features: {feature_names}")

    # 7. Train all models
    print("\n[7/7] Training models...")
    trained_models = train_all_models(X_train_processed, y_train)

    # Save everything
    print("\n--- Saving artifacts ---")
    os.makedirs('models/', exist_ok=True)
    os.makedirs('data/processed/', exist_ok=True)

    for name, model in trained_models.items():
        path = save_model(model, name)
        print(f"  Saved {name} -> {path}")

    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("  Saved preprocessor -> models/preprocessor.joblib")

    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=4)
    print("  Saved feature names -> models/feature_names.json")

    # Save processed data splits
    X_train_processed.to_csv('data/processed/X_train.csv', index=False)
    X_val_processed.to_csv('data/processed/X_val.csv', index=False)
    X_test_processed.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print("  Saved processed data splits -> data/processed/")

    # Save best model as the production model (will be updated after evaluation)
    # For now, save XGBoost as default
    joblib.dump(trained_models['XGBoost'], 'models/model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("  Saved production model -> models/model.pkl")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Models trained: {list(trained_models.keys())}")
    print("=" * 60)
