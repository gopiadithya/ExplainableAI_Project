import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib

class TestPrediction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = joblib.load('models/best_model.joblib')
        self.preprocessor = joblib.load('models/preprocessor.joblib')
        import json
        with open('models/feature_names.json') as f:
            self.feature_names = json.load(f)
    
    def _make_sample_input(self):
        """Create a sample input DataFrame."""
        data = {
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100],
            'Type_H': [False],
            'Type_L': [True],
            'Type_M': [False]
        }
        return pd.DataFrame(data)
    
    def test_model_loads(self):
        assert self.model is not None
    
    def test_preprocessor_loads(self):
        assert self.preprocessor is not None
    
    def test_prediction_output_shape(self):
        df = self._make_sample_input()
        processed = self.preprocessor.transform(df)
        pred = self.model.predict(processed)
        assert len(pred) == 1
        assert pred[0] in [0, 1]
    
    def test_probability_range(self):
        df = self._make_sample_input()
        processed = self.preprocessor.transform(df)
        proba = self.model.predict_proba(processed)
        assert proba.shape == (1, 2)
        assert 0 <= proba[0][0] <= 1
        assert 0 <= proba[0][1] <= 1
        assert abs(proba[0][0] + proba[0][1] - 1.0) < 1e-6
    
    def test_feature_names_match(self):
        assert len(self.feature_names) == 8
        assert 'Type_H' in self.feature_names
