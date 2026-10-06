import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.preprocessing import (
    RANDOM_STATE, TARGET_COL, LEAK_COLS, ID_COLS,
    remove_leakage_columns, encode_categorical,
    get_feature_target_split, create_train_test_split,
    build_preprocessing_pipeline
)

class TestDataLoading:
    def test_load_raw_data(self):
        df = load_raw_data()
        assert df.shape == (10000, 14)
        assert 'Machine failure' in df.columns
    
    def test_no_missing_values(self):
        df = load_raw_data()
        assert df.isnull().sum().sum() == 0

class TestPreprocessing:
    def test_remove_leakage_columns(self):
        df = load_raw_data()
        df_clean, removed = remove_leakage_columns(df)
        for col in LEAK_COLS:
            assert col not in df_clean.columns
        for col in ID_COLS:
            assert col not in df_clean.columns
        assert TARGET_COL in df_clean.columns
    
    def test_encode_categorical(self):
        df = load_raw_data()
        df_clean, _ = remove_leakage_columns(df)
        df_enc = encode_categorical(df_clean)
        assert 'Type' not in df_enc.columns
        assert 'Type_H' in df_enc.columns
        assert 'Type_L' in df_enc.columns
        assert 'Type_M' in df_enc.columns
    
    def test_feature_target_split(self):
        df = load_raw_data()
        df_clean, _ = remove_leakage_columns(df)
        df_enc = encode_categorical(df_clean)
        X, y = get_feature_target_split(df_enc)
        assert TARGET_COL not in X.columns
        assert len(y) == len(X)
        assert set(y.unique()) == {0, 1}
    
    def test_no_target_leakage(self):
        """Critical test: ensure no leakage columns in features."""
        df = load_raw_data()
        df_clean, _ = remove_leakage_columns(df)
        df_enc = encode_categorical(df_clean)
        X, y = get_feature_target_split(df_enc)
        for col in LEAK_COLS:
            assert col not in X.columns, f"Target leakage: {col} found in features!"
    
    def test_train_test_split_stratified(self):
        df = load_raw_data()
        df_clean, _ = remove_leakage_columns(df)
        df_enc = encode_categorical(df_clean)
        X, y = get_feature_target_split(df_enc)
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
        # Check sizes
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        # Check stratification (failure rate should be similar across splits)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02  # Within 2%
    
    def test_preprocessing_pipeline(self):
        preprocessor = build_preprocessing_pipeline()
        assert preprocessor is not None
