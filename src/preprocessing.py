import os
import json
import re
import pandas as pd
import joblib
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_raw_data

RANDOM_STATE = 42
TARGET_COL = 'Machine failure'
LEAK_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
ID_COLS = ['UDI', 'Product ID']
CATEGORICAL_COLS = ['Type']
NUMERICAL_COLS = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """Sanitize column names for XGBoost compatibility.
    XGBoost does not allow [, ], or < in feature names.
    """
    return [re.sub(r'[\[\]<]', lambda m: '(' if m.group() == '[' else ')' if m.group() == ']' else '_', col) for col in columns]


def get_sanitized_feature_names() -> List[str]:
    """Returns sanitized feature names (after preprocessing)."""
    raw_names = NUMERICAL_COLS + ['Type_H', 'Type_L', 'Type_M']
    return sanitize_feature_names(raw_names)

def remove_leakage_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Removes leakage and ID columns from the dataframe.
    """
    df_clean = df.copy()
    removed_info = {}
    
    cols_to_drop = []
    for col in LEAK_COLS:
        if col in df_clean.columns:
            cols_to_drop.append(col)
            removed_info[col] = "Target leakage"
            
    for col in ID_COLS:
        if col in df_clean.columns:
            cols_to_drop.append(col)
            removed_info[col] = "Identifier, not predictive"
            
    df_clean = df_clean.drop(columns=cols_to_drop)
    return df_clean, removed_info

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the categorical features.
    """
    df_encoded = df.copy()
    if 'Type' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['Type'], drop_first=False)
        # Ensure all types are present
        for col in ['Type_H', 'Type_L', 'Type_M']:
            if col not in df_encoded.columns:
                df_encoded[col] = False
    return df_encoded

def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataframe into features (X) and target (y).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe")
        
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def create_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs stratified train/val/test split.
    val_size is the proportion of the original dataset.
    """
    # First split into train and temp (val + test)
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Now split temp into val and test
    test_ratio = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds a scikit-learn preprocessing pipeline.
    """
    encoded_cat_cols = ['Type_H', 'Type_L', 'Type_M']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', 'passthrough', encoded_cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def preprocess_and_save(output_dir: str = 'data/processed/') -> None:
    """
    Runs the full pipeline and saves the outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    df = load_raw_data()
    
    # 2. Clean data
    df_clean, removed_info = remove_leakage_columns(df)
    
    # 3. Encode categoricals
    df_encoded = encode_categorical(df_clean)
    
    # 4. Split X/y
    X, y = get_feature_target_split(df_encoded)
    
    # 5. Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    
    # 6. Fit preprocessor
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=NUMERICAL_COLS + ['Type_H', 'Type_L', 'Type_M'], index=X_train.index)
    X_val_processed = pd.DataFrame(preprocessor.transform(X_val), columns=NUMERICAL_COLS + ['Type_H', 'Type_L', 'Type_M'], index=X_val.index)
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=NUMERICAL_COLS + ['Type_H', 'Type_L', 'Type_M'], index=X_test.index)
    
    # Save outputs
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    
    X_train_processed.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_val_processed.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    X_test_processed.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    feature_names = NUMERICAL_COLS + ['Type_H', 'Type_L', 'Type_M']
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=4)
        
    report = {
        "removed_columns": removed_info,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    }
    with open(os.path.join(output_dir, 'preprocessing_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
