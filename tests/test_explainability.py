import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
import shap
import json

class TestExplainability:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = joblib.load('models/best_model.joblib')
        self.preprocessor = joblib.load('models/preprocessor.joblib')
        with open('models/feature_names.json') as f:
            self.feature_names = json.load(f)
        self.X_test = pd.read_csv('data/processed/X_test.csv')
        self.explainer = shap.TreeExplainer(self.model)
    
    def test_shap_explainer_creation(self):
        assert self.explainer is not None
    
    def test_shap_values_computation(self):
        sample = self.X_test.head(5)
        shap_values = self.explainer(sample)
        # Should produce values for each feature
        assert shap_values.values.shape[1] == len(self.feature_names)
    
    def test_shap_values_for_single_instance(self):
        sample = self.X_test.head(1)
        shap_values = self.explainer(sample)
        values = shap_values.values
        # Handle 3D case (binary classification)
        if len(values.shape) == 3:
            values = values[:, :, 1]
        assert values.shape == (1, len(self.feature_names))
    
    def test_feature_contributions(self):
        from src.explain import get_feature_contributions, compute_shap_values
        sample = self.X_test.head(1)
        shap_values = compute_shap_values(self.explainer, sample, self.feature_names)
        contributions = get_feature_contributions(shap_values, self.feature_names, index=0)
        assert len(contributions) == len(self.feature_names)
        for feat, val, direction in contributions:
            assert feat in self.feature_names
            assert direction in ['increases risk', 'decreases risk']
    
    def test_risk_categories(self):
        from src.explain import get_risk_category
        assert get_risk_category(0.1) == 'Low Risk'
        assert get_risk_category(0.4) == 'Medium Risk'
        assert get_risk_category(0.8) == 'High Risk'
    
    def test_natural_language_explanation(self):
        from src.explain import generate_natural_language_explanation
        contributions = [('Torque (Nm)', 0.5, 'increases risk'), ('Tool wear (min)', -0.3, 'decreases risk')]
        explanation = generate_natural_language_explanation(0.75, contributions)
        assert 'HIGH' in explanation
        assert 'Torque' in explanation
    
    def test_model_artifacts_exist(self):
        assert os.path.exists('models/best_model.joblib')
        assert os.path.exists('models/preprocessor.joblib')
        assert os.path.exists('models/feature_names.json')
