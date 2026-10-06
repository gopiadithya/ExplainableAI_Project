# X-Maintain: Academic Project Review 2 Dossier
**Course:** Explainable Artificial Intelligence (XAI)  
**Project Title:** X-Maintain — Explainable AI for Predictive Maintenance  
**Review Date:** Review 2 — October 2026  
**Status:** 100% Fully Implemented, Validated, and Reproducible  

---

## 1. Executive Review Summary

This dossier provides a comprehensive defense summary for **Review 2**. All requirements have been implemented programmatically using real data, strict leakage guards, reproducible seeds, multiple machine learning architectures, explainability frameworks, and interactive demonstration interfaces.

---

## 2. Dataset & Target Leakage Prevention Verification

### Dataset
- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** UCI Machine Learning Repository
- **Total Records:** 10,000 milling operations
- **Target Variable:** `Machine failure` (0 = Safe, 1 = Breakdown)
- **Class Breakdown:**
  - Class 0 (No Failure): 9,661 samples (96.61%)
  - Class 1 (Failure): 339 samples (3.39%)
  - Imbalance Ratio: ~28.5 : 1

### Academic Target Leakage Audit
The raw AI4I dataset contains five specific failure modes:
1. `TWF` (Tool Wear Failure)
2. `HDF` (Heat Dissipation Failure)
3. `PWF` (Power Failure)
4. `OSF` (Overstrain Failure)
5. `RNF` (Random Failure)

> **Defense Rationale**: `Machine failure` is a direct boolean logical function of these failure mode columns: $\text{Failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$. Retaining these columns as predictive features causes target leakage, artificially driving model metrics to near 100% accuracy while making the model useless for proactive maintenance.
>
> In **X-Maintain**, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are strictly dropped prior to any preprocessing. In addition, arbitrary serial columns (`UDI`, `Product ID`) are dropped.
>
> **Verification**: Verified programmatically in unit tests via `tests/test_preprocessing.py::TestPreprocessing::test_no_target_leakage`.

---

## 3. End-to-End Pipeline & Reproducibility Audit

1. **Stratified Split**:
   - Training Set: 6,999 samples (70%)
   - Validation Set: 1,000 samples (10%)
   - Test Set: 2,001 samples (20%) — 68 failure cases
   - Fixed seed: `RANDOM_STATE = 42`
2. **Preprocessing Pipeline**:
   - `StandardScaler` fitted solely on training numerical data
   - One-hot encoding for product variant `Type` (`Type_H`, `Type_L`, `Type_M`)
   - Feature name sanitization for XGBoost DMatrix compliance
   - Pipeline serialized to `models/preprocessor.joblib`

---

## 4. Empirical Evaluation & Research Experiments

All reported metrics are measured on the **held-out test split ($N = 2,001$)**:

### Comprehensive Experiment Summary Table

| Experiment Index | Model Architecture & Configuration | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|---|
| **Exp 4b** | XGBoost + SMOTE (Applied to Train Split Only) | 97.85% | 0.6506 | **0.7941** | 0.7152 | 0.9620 | **0.7996** | **0.8208** |
| **Exp 4c** | XGBoost + Val-Tuned Threshold ($\tau^* = 0.6123$) | **98.35%** | **0.7692** | 0.7353 | **0.7519** | **0.9724** | 0.7934 | 0.8114 |
| **Exp 4a** | XGBoost + Cost-Sensitive (`scale_pos_weight = 28.53`) | 98.20% | 0.7353 | 0.7353 | 0.7353 | **0.9724** | 0.7934 | 0.8064 |
| **Exp 3** | XGBoost Standard (Unweighted Logloss) | 98.30% | 0.8036 | 0.6618 | 0.7258 | 0.9653 | 0.7874 | 0.7720 |
| **Exp 1** | Logistic Regression (Baseline, Balanced) | 83.61% | 0.1467 | **0.7941** | 0.2477 | 0.8949 | 0.4187 | 0.6604 |
| **Exp 2** | Random Forest (Balanced Bagging, 200 Trees) | 97.75% | 0.8710 | 0.3971 | 0.5455 | 0.9556 | 0.6930 | 0.6091 |

### Model Selection Rationale
- **Objective Function**: $\text{Utility} = 0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$
- While Logistic Regression captures 79.41% of failures, its precision is only 14.67% (over 300 false alarms), leading to severe alarm fatigue.
- Random Forest misses 60.29% of machine failures (Recall = 39.71%), risking catastrophic factory downtime.
- **XGBoost (Cost-Sensitive)** achieves an operational balance with **73.53% Recall**, **73.53% Precision**, **0.9724 ROC-AUC**, and **0.7934 PR-AUC**.

---

## 5. Explainable AI (XAI) Implementation Details

### Framework: TreeSHAP (TreeExplainer)
- TreeSHAP computes game-theoretic Shapley attributions in $\mathcal{O}(TLD^2)$ polynomial time.
- Satisfies local accuracy, missingness, and consistency axioms.

### Global Attributions
- **Mean Absolute SHAP Ranking**:
  1. `Torque (Nm)`: $3.740$ (Extreme torsional shear accelerates mechanical breakdown)
  2. `Tool wear (min)`: $3.057$ (Cumulative abrasion causes dimensional loss & chatter)
  3. `Rotational speed (rpm)`: $2.070$ (Thermal friction & centrifugal stresses)
  4. `Air temperature (K)`: $2.005$ (Environmental heat reduces dissipation gradient)
  5. `Process temperature (K)`: $0.988$ (Governs $\Delta T$ dissipation efficiency)
  6. `Type_L`: $0.196$ (Low-tier product variant failure bias)
  7. `Type_M`: $0.174$ (Medium-tier variant)
  8. `Type_H`: $0.088$ (High-tier heavy duty variant)

### Local Attribution & Waterfall Decomposition
- For any individual machine status, TreeSHAP decomposes the model output:
  $$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
- Visualized via interactive waterfall plots in the Streamlit application.

### Deterministic Natural Language Generator
- Avoids LLM hallucinations by mapping directly from computed SHAP values to structured text:
  *"The model predicts a HIGH failure risk (probability: 0.88). The strongest factors contributing to this prediction are: Torque (Nm) (increases risk), Tool wear (min) (increases risk). Factors decreasing risk: Rotational speed (rpm) (decreases risk)."*

---

## 6. Model-Based What-If Analysis & Non-Causal Grounding

The dashboard includes a counterfactual parameter sensitivity tool.

> **Academic Rigor Notice**:  
> What-if parameter perturbation is strictly labeled as **model-based sensitivity analysis**, **NOT causal intervention**. Changing a slider value observes the model's learned response manifold; it does not constitute a physical randomized control trial or guarantee identical thermodynamic behavior.

---

## 7. Viva Voce & Presentation Q&A Preparation

### Q1: Why did you drop the failure mode columns (TWF, HDF, PWF, OSF, RNF)?
**Answer:** Retaining them would cause target leakage. In this dataset, `Machine failure` is a logical OR composite of these five specific modes. Including them gives the model access to the label, resulting in 99.9% artificial accuracy and rendering the model useless for true predictive maintenance.

### Q2: Why did you not rely solely on accuracy for model evaluation?
**Answer:** With a 96.61% negative class majority, a degenerate model predicting "No Failure" for every sample achieves 96.61% accuracy while failing to detect 100% of machine breakdowns. In manufacturing, undetected failures cause major equipment damage and production stoppages, whereas false alarms only incur a minor inspection cost. Therefore, failure-class Recall, F1, and PR-AUC are far more appropriate metrics.

### Q3: Why is TreeSHAP preferred over LIME or KernelSHAP for tree ensembles?
**Answer:** KernelSHAP and LIME rely on sampling and linear local surrogate approximations, which can be computationally slow and introduce sampling variance. TreeSHAP exploits the internal decision paths of trees to compute exact Shapley values in polynomial time $\mathcal{O}(TLD^2)$, guaranteeing local accuracy and consistency.

### Q4: Does the What-If analysis prove that lowering tool wear prevents failure?
**Answer:** No. What-If analysis evaluates the model's learned conditional response surface. While it provides useful operational guidance, it represents observational association rather than physical causal invariance.

---

## 8. Verification Commands for Live Demonstration

```powershell
# 1. Run all 20 automated tests
python -m pytest tests/ -v

# 2. Retrain all models and serialize pipelines
python -m src.train

# 3. Evaluate models and output benchmark table
python -m src.evaluate

# 4. Run the research imbalance experiments
python -m src.experiments

# 5. Launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```
