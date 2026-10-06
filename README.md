# X-Maintain: Explainable AI for Predictive Maintenance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.49.1-brightgreen.svg)](https://shap.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-orange.svg)](https://xgboost.readthedocs.io/)

An end-to-end Explainable AI (XAI) system for industrial predictive maintenance, built for an undergraduate Explainable AI course project. The system predicts machine failure using multivariate sensor data and explains **WHY** each prediction was made using SHapley Additive exPlanations (SHAP).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Motivation](#3-motivation)
4. [Primary Objectives](#4-primary-objectives)
5. [Dataset Description](#5-dataset-description)
6. [Dataset Limitations & Critical Leakage Requirements](#6-dataset-limitations--critical-leakage-requirements)
7. [Data Preprocessing Pipeline](#7-data-preprocessing-pipeline)
8. [Feature Engineering & Encoding](#8-feature-engineering--encoding)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Class Imbalance Handling](#10-class-imbalance-handling)
11. [Model Evaluation & Selection Criterion](#11-model-evaluation--selection-criterion)
12. [Experimental Results](#12-experimental-results)
13. [Explainable AI (XAI) Methodology](#13-explainable-ai-xai-methodology)
14. [Global & Local SHAP Explanations](#14-global--local-shap-explanations)
15. [Deterministic Natural-Language Explanations](#15-deterministic-natural-language-explanations)
16. [Model-Based What-If Analysis](#16-model-based-what-if-analysis)
17. [System Architecture & Workflow](#17-system-architecture--workflow)
18. [Installation & Setup](#18-installation--setup)
19. [Reproducibility Guide](#19-reproducibility-guide)
20. [Streamlit Dashboard Guide](#20-streamlit-dashboard-guide)
21. [Automated Testing](#21-automated-testing)
22. [Research Limitations](#22-research-limitations)
23. [Future Enhancements](#23-future-enhancements)
24. [References](#24-references)

---

## 1. Project Overview

Predictive maintenance (PdM) leverages machine sensor data to anticipate equipment failures before they cause costly unplanned downtime. However, standard machine learning models operate as "black boxes," providing predictions without actionable rationale. In mission-critical industrial manufacturing, operators cannot risk shutting down a production line or replacing expensive components based solely on an opaque probability score.

**X-Maintain** bridges this trust gap by combining high-performance gradient boosting with TreeSHAP-based post-hoc local and global explanations, deterministic natural-language summaries, and a model-based what-if sensitivity simulator.

---

## 2. Problem Statement

Given a set of operational parameters from a milling machine—comprising product variant, air temperature, process temperature, rotational speed, torque, and accumulated tool wear—determine whether the machine is at imminent risk of failure, and produce an interpretable, mathematically sound explanation of the specific physical sensor contributions leading to that decision.

---

## 3. Motivation

1. **Economic Impact**: Unplanned downtime costs industrial manufacturers an estimated \$50 billion annually.
2. **Asymmetric Error Costs**: A false alarm incurs a brief inspection cost (\$100-\$500), whereas an undetected failure causes catastrophic spindle damage, collateral tooling destruction, and emergency outages (\$10,000-\$100,000+). Therefore, **Recall on the Failure class is paramount**.
3. **The XAI Imperative**: Machine operators require actionable diagnosis (e.g., *"Tool wear has exceeded safe limits under high torque"*), not just an alert.

---

## 4. Primary Objectives

- Build a target-leakage-free preprocessing and modeling pipeline.
- Formulate and compare three distinct algorithmic families (Logistic Regression, Random Forest, XGBoost) with proper class imbalance compensation.
- Optimize model selection using a weighted domain utility function: $\text{Score} = 0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$.
- Implement SHAP TreeExplainer for mathematically consistent local attribution and global feature ranking.
- Deploy an interactive Streamlit dashboard featuring deterministic natural language reporting and interactive parameter sensitivity analysis.
- Maintain 100% test coverage over core pipeline transformations, prediction ranges, and XAI outputs.

---

## 5. Dataset Description

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- **Source**: UCI Machine Learning Repository (Matheader et al., 2020)
- **URL**: `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`
- **Total Records**: 10,000 instances
- **Features**: 14 raw columns
- **Target**: `Machine failure` (Binary: 0 = No Failure, 1 = Failure)
- **Class Distribution**:
  - No Failure (0): 9,661 samples (96.61%)
  - Failure (1): 339 samples (3.39%)
  - **Severe Imbalance Ratio**: ~28.5 : 1

### Feature Breakdown

| Feature Name | Type | Unit | Description | Range in Data |
|---|---|---|---|---|
| `UDI` | Integer | - | Unique row identifier | 1 - 10000 |
| `Product ID` | String | - | Product serial number (`L/M/H` + id) | Categorical |
| `Type` | Categorical | - | Quality variant (`L`=50%, `M`=30%, `H`=20%) | L, M, H |
| `Air temperature [K]` | Float | Kelvin | Ambient air temperature | 295.3 - 304.5 K |
| `Process temperature [K]` | Float | Kelvin | Process temperature during operation | 305.7 - 313.8 K |
| `Rotational speed [rpm]` | Integer | RPM | Spindle speed | 1168 - 2886 RPM |
| `Torque [Nm]` | Float | Nm | Spindle torque | 3.8 - 76.6 Nm |
| `Tool wear [min]` | Integer | Minutes | Cumulative cutting tool usage | 0 - 253 min |
| `Machine failure` | Binary | - | **TARGET VARIABLE** | 0 or 1 |
| `TWF` | Binary | - | Tool Wear Failure mode | 0 or 1 |
| `HDF` | Binary | - | Heat Dissipation Failure mode | 0 or 1 |
| `PWF` | Binary | - | Power Failure mode | 0 or 1 |
| `OSF` | Binary | - | Overstrain Failure mode | 0 or 1 |
| `RNF` | Binary | - | Random Failure mode | 0 or 1 |

---

## 6. Dataset Limitations & Critical Leakage Requirements

### Target Leakage Prevention (Crucial Academic Requirement)
The dataset includes five failure-mode indicators: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`. If any of these are included as input features, the model trivially learns that $\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$, yielding 99.9% accuracy with zero real-world predictive validity.

**Strict Exclusion**:
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are strictly dropped before preprocessing.
- `UDI` (arbitrary row index) is dropped.
- `Product ID` (redundant with `Type` and high-cardinality serial numbers) is dropped.

All exclusions are verified via automated assertions in `tests/test_preprocessing.py`.

---

## 7. Data Preprocessing Pipeline

The preprocessing workflow is entirely reproducible and follows scikit-learn best practices:

1. **Stratified Splitting**:
   - `test_size = 0.2` (2,001 samples)
   - `val_size = 0.1` (1,000 samples)
   - `train_size = 0.7` (6,999 samples)
   - `stratify=y` ensures the 3.39% failure incidence is maintained across splits.
2. **Numerical Standardization**:
   - `StandardScaler` fitted **only on the training split** to prevent data snooping/leakage.
   - Scaled features: Air temp, Process temp, Rotational speed, Torque, Tool wear.
3. **Categorical Encoding**:
   - One-hot encoding of `Type` into `Type_H`, `Type_L`, `Type_M`.
   - Passthrough in `ColumnTransformer`.
4. **Name Sanitization**:
   - Feature names are sanitized from brackets `[...]` to parentheses `(...)` for strict compatibility with modern XGBoost DMatrix specifications.

---

## 8. Feature Engineering & Encoding

The final preprocessed feature set consists of 8 columns:

1. `Air temperature (K)`
2. `Process temperature (K)`
3. `Rotational speed (rpm)`
4. `Torque (Nm)`
5. `Tool wear (min)`
6. `Type_H` (High-quality variant indicator)
7. `Type_L` (Low-quality variant indicator)
8. `Type_M` (Medium-quality variant indicator)

---

## 9. Machine Learning Models

Three distinct model families were benchmarked:

### 1. Baseline: Logistic Regression
- **Algorithm**: Regularized linear classifier with L2 penalty.
- **Config**: `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- **Purpose**: Establishes linear separability performance baseline.

### 2. Non-Linear Ensemble: Random Forest
- **Algorithm**: Bagged decision trees with random subspace feature sampling.
- **Config**: `n_estimators=200`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42`.
- **Purpose**: Evaluates non-linear bagging on imbalanced industrial tabular data.

### 3. Gradient Boosted Trees: XGBoost
- **Algorithm**: Second-order gradient-boosted decision trees.
- **Config**: `n_estimators=200`, `scale_pos_weight=28.53`, `eval_metric='logloss'`, `random_state=42`.
- **Purpose**: SOTA gradient boosting with dynamic positive class re-weighting.

---

## 10. Class Imbalance Handling

Due to the 28.5:1 negative-to-positive ratio, unweighted standard accuracy optimization results in a degenerate model predicting all zeros (96.6% accuracy, 0% recall).

### Imbalance Strategy Comparison:
1. **SMOTE (Synthetic Minority Over-sampling)**: Evaluated; risks generating unphysical sensor combinations in multivariate thermodynamic space.
2. **Cost-Sensitive Weighting (`class_weight='balanced'` / `scale_pos_weight`)**:
   - **Logistic Regression**: `class_weight='balanced'` assigns $\frac{N}{2 \times N_k}$ weight.
   - **Random Forest**: Balanced sub-sample trees.
   - **XGBoost**: $\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{6762}{237} \approx 28.53$.

Cost-sensitive weighting directly addresses asymmetric loss while preserving the empirical multivariate manifold.

---

## 11. Model Evaluation & Selection Criterion

### Why Accuracy is Insufficient
A naive "always predict safe" classifier achieves 96.6% accuracy but misses 100% of failures, leading to catastrophic factory damage.

### Evaluation Metrics
- **Recall (Class 1)**: Primary safety metric. Minimizes False Negatives.
- **Precision (Class 1)**: Operational efficiency metric. Prevents alarm fatigue.
- **F1-Score**: Harmonic mean balancing precision and recall.
- **ROC-AUC**: Discrimination threshold invariance.
- **PR-AUC (Average Precision)**: Informative metric under severe imbalance.

### Selection Criterion: Weighted Utility Score
$$\text{Utility} = 0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$$

---

## 12. Experimental Results

All results reported below were generated programmatically from the held-out test split ($N = 2,001$ samples, 68 failure cases):

### Primary Benchmark

| Model | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 83.61% | 0.1467 | **0.7941** | 0.2477 | 0.8949 | 0.4187 | 0.6604 |
| **Random Forest** | 97.75% | **0.8710** | 0.3971 | 0.5455 | 0.9556 | 0.6930 | 0.6091 |
| **XGBoost (Selected)** | **98.20%** | 0.7353 | 0.7353 | **0.7353** | **0.9724** | **0.7934** | **0.8064** |

### Research-Style Experiments: Imbalance Mitigation Study (`src/experiments.py`)

To satisfy Section 8 and Section 20 requirements, we conducted a systematic empirical comparison of class imbalance mitigation strategies on the test split ($N = 2,001$):

| Experiment Configuration | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|
| **Exp 4b: XGBoost (SMOTE on Train Only)** | 97.85% | 0.6506 | **0.7941** | 0.7152 | 0.9620 | **0.7996** | **0.8208** |
| **Exp 4c: XGBoost (Threshold Tuned = 0.6123)** | **98.35%** | **0.7692** | 0.7353 | **0.7519** | **0.9724** | 0.7934 | 0.8114 |
| **Exp 4a: XGBoost (Cost-Sensitive `scale_pos_weight`)** | 98.20% | 0.7353 | 0.7353 | 0.7353 | **0.9724** | 0.7934 | 0.8064 |
| **Exp 3: XGBoost (Unweighted / Standard)** | 98.30% | 0.8036 | 0.6618 | 0.7258 | 0.9653 | 0.7874 | 0.7720 |
| **Exp 1: Logistic Regression (Balanced Baseline)** | 83.61% | 0.1467 | **0.7941** | 0.2477 | 0.8949 | 0.4187 | 0.6604 |
| **Exp 2: Random Forest (Balanced Bagging)** | 97.75% | 0.8710 | 0.3971 | 0.5455 | 0.9556 | 0.6930 | 0.6091 |

### Key Experimental Insights:
1. **Unweighted XGBoost (Exp 3)** exhibits a substantial recall deficit ($66.18\%$), missing over a third of actual machine breakdowns due to symmetric logloss penalties.
2. **Cost-Sensitive Weighting (Exp 4a)** balances failure recall and precision symmetrically at $73.53\%$, preventing alarm fatigue while capturing nearly three-quarters of breakdowns without synthetic data distortion.
3. **SMOTE on Training Data (Exp 4b)** attains the highest recall ($79.41\%$), but causes a $8.5\%$ drop in precision ($65.06\%$) due to interpolated synthetic feature vectors along non-convex thermodynamic boundaries.
4. **Decision Threshold Tuning (Exp 4c)** shifts the decision threshold on validation data to $\tau^* = 0.6123$, yielding the highest overall F1-score ($0.7519$) and precision ($76.92\%$).

---

## 13. Explainable AI (XAI) Methodology

We implement **SHAP (SHapley Additive exPlanations)** based on cooperative game theory (Lundberg & Lee, 2017).

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

### TreeSHAP
For our selected XGBoost model, we employ **TreeExplainer** (Lundberg et al., 2020), which computes exact Shapley values in polynomial time $\mathcal{O}(T L D^2)$ instead of exponential sampling time.

---

## 14. Global & Local SHAP Explanations

### Global Feature Importance Ranking

Based on mean absolute SHAP values across the test dataset:

| Rank | Feature | Mean \|SHAP\| Value | Physical Interpretation |
|---|---|---|---|
| 1 | **Torque (Nm)** | **3.740** | High torque induces mechanical overload & power strain |
| 2 | **Tool wear (min)** | **3.057** | Cumulative tool friction degrades spindle cutting integrity |
| 3 | **Rotational speed (rpm)** | **2.070** | Extreme RPM causes thermal buildup and centrifugal stress |
| 4 | **Air temperature (K)** | **2.005** | Ambient heat hampers convection and heat dissipation |
| 5 | **Process temperature (K)** | **0.988** | Temperature delta $(\Delta T)$ governs thermal dissipation failure |
| 6 | **Type_L** | **0.196** | Low quality variant has higher defect propensity |
| 7 | **Type_M** | **0.174** | Medium quality baseline variant |
| 8 | **Type_H** | **0.088** | High quality heavy-duty variant (least failure-prone) |

### Visual Artifacts Generated
All artifacts are generated programmatically and saved in `artifacts/figures/`:
- `artifacts/figures/target_distribution.png`: Class balance breakdown.
- `artifacts/figures/numerical_distributions.png`: Feature distributions with sample means.
- `artifacts/figures/boxplots.png`: Outlier inspection across operational variables.
- `artifacts/figures/correlation_heatmap.png`: Inter-feature Pearson correlation.
- `artifacts/figures/failure_by_type.png`: Empirical failure rate across product variants.
- `artifacts/figures/feature_by_failure.png`: Comparative sensor histograms by failure state.
- `artifacts/figures/cm_xgboost.png`, `cm_random_forest.png`, `cm_logistic_regression.png`: Confusion matrices.
- `artifacts/figures/roc_curves.png`: Multi-model ROC comparison.
- `artifacts/figures/pr_curves.png`: Precision-Recall curves.
- `artifacts/figures/shap_global_importance.png`: Mean $|SHAP|$ bar summary.
- `artifacts/figures/shap_beeswarm.png`: Beeswarm plot revealing directional feature impact.
- `artifacts/figures/shap_waterfall.png`: Sample prediction waterfall decomposition.
- `artifacts/figures/shap_dependence_*.png`: Non-linear feature interaction plots.
- `artifacts/figures/architecture_diagram.png`: Comprehensive technical system architecture.
- `artifacts/figures/workflow_diagram.png`: Academic workflow schematic.

---

## 15. Deterministic Natural-Language Explanations

To ensure strict reproducibility without hallucinations, X-Maintain avoids ungrounded LLM generation for core risk explanations. Instead, explanations are generated using a deterministic rule engine mapped directly to computed SHAP values:

```python
# Sample Output:
"The model predicts a HIGH failure risk (probability: 0.82). 
 The strongest factors contributing to this prediction are: 
 Torque (Nm) (increases risk), Tool wear (min) (increases risk). 
 Factors decreasing risk: Rotational speed (rpm) (decreases risk)."
```

---

## 16. Model-Based What-If Analysis

The system allows operators to explore counterfactual parameter sensitivity:

> **Important Academic Disclaimer**:  
> What-if analysis is **model-based sensitivity analysis**, **NOT** empirical causal intervention. Lowering tool wear in the model simulator reduces model-estimated risk, but does not guarantee immunity from unmodeled physical failure modes.

### Example Scenario:
- **Baseline**: Tool wear = 210 min, Torque = 65 Nm $\rightarrow$ **Failure Probability = 88.4% (High Risk)**
- **What-If Adjustment**: Replace tool (Tool wear = 15 min) $\rightarrow$ **Failure Probability = 4.2% (Low Risk)**
- **Net Impact**: **-84.2% Risk Reduction**

---

## 17. System Architecture & Workflow

```
[ Industrial Sensor Data ]
           │
           ▼
[ Data Preprocessing Pipeline ]
    ├── Target Leakage Purge (Drop TWF, HDF, PWF, OSF, RNF, UDI, Product ID)
    ├── Stratified Split (70% Train / 10% Val / 20% Test)
    ├── One-Hot Categorical Encoding (Type -> Type_H, Type_L, Type_M)
    └── StandardScaler (Numerical variables, Train-fitted)
           │
           ▼
[ Multi-Model Training & Benchmarking ]
    ├── Logistic Regression (Balanced Baseline)
    ├── Random Forest (Balanced Ensemble)
    └── XGBoost (scale_pos_weight = 28.53)
           │
           ▼
[ Weighted Utility Model Selection ]
    └── Best: XGBoost (Utility = 0.8064, ROC-AUC = 0.9724, Recall = 0.7353)
           │
           ▼
[ Explainable AI Layer (TreeSHAP) ]
    ├── Global Importance (Bar / Beeswarm / Dependence)
    ├── Local Waterfall Attribution
    └── Deterministic Natural Language Summary
           │
           ▼
[ Production Interfaces ]
    ├── Interactive Streamlit Application (app/streamlit_app.py)
    └── What-If Scenario Simulator
```

---

## 18. Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/your-username/X-Maintain.git
cd X-Maintain

# 2. Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 19. Reproducibility Guide

To reproduce all experiments, models, and artifacts from scratch:

```bash
# Step 1: Run Preprocessing & Training
python -m src.train

# Step 2: Run Multi-Model Evaluation & Selection
python -m src.evaluate

# Step 3: Generate Exploratory Data Analysis Plots
python -m src.eda

# Step 4: Run Automated Verification Suite
python -m pytest tests/ -v

# Step 5: Launch Streamlit Dashboard
streamlit run app/streamlit_app.py
```

---

## 20. Streamlit Dashboard Guide

Launch the UI:
```bash
streamlit run app/streamlit_app.py
```

The application provides 6 distinct views:
1. **Home**: Project motivation, problem statement, and key architectural highlights.
2. **Prediction**: Interactive sliders for operational parameters with real-time risk assessment, probability gauge, contributing factor table, and local SHAP waterfall breakdown.
3. **Explainability**: Global feature importance comparisons, beeswarm summary distribution, and interactive test case explorer.
4. **What-If Analysis**: Side-by-side counterfactual simulator comparing original and adjusted operating conditions.
5. **Model Performance**: Full evaluation benchmark table, confusion matrices, ROC curves, and PR curves.
6. **Dataset Insights**: Interactive EDA explorer showing raw distributions, correlations, and class balance.

---

## 21. Automated Testing

The project includes an automated test suite with 20 passing unit tests:

```bash
python -m pytest tests/ -v
```

### Test Coverage Summary:
- `tests/test_preprocessing.py`: Validates raw dataset ingestion, zero missing values, leakage column exclusion, one-hot encoding shape, stratified split ratios, and pipeline instantiation.
- `tests/test_prediction.py`: Validates artifact serialization, single-instance prediction shapes, valid probability bounds $[0.0, 1.0]$, and feature vector alignment.
- `tests/test_explainability.py`: Validates TreeExplainer initialization, 2D/3D SHAP tensor slicing, directional attribution consistency, risk category mapping, and natural language synthesis.

---

## 22. Research Limitations

1. **Synthetic Nature of Dataset**: The AI4I 2020 dataset was generated using physical simulation models; true industrial sensor logs exhibit non-stationary sensor drift, packet loss, and sensor noise not fully modeled here.
2. **Cross-Machine Generalizability**: The model is calibrated for milling machines operating under specific thermal environments; deployment on distinct kinematics (e.g., lathes, presses) requires transfer learning.
3. **Non-Causal Attribute Interpretation**: SHAP reflects observational feature associations within model logic; it cannot prove counterfactual physical causality.

---

## 23. Future Enhancements

- **Time-Series Deep Learning**: Temporal Convolutional Networks (TCN) or LSTMs for remaining useful life (RUL) estimation using sequential vibration streams.
- **Multimodal Visual Inspection**: Fusion of sensor telemetry with MVTec AD visual inspection imagery.
- **Continuous Edge Telemetry**: Model quantization and deployment onto embedded microcontrollers (Raspberry Pi / NVIDIA Jetson) using ONNX runtime.
- **Automated Drift Detection**: Evidentiary data drift monitoring via Kolmogorov-Smirnov statistical testing.

---

## 24. References

1. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. In Third International Conference on Artificial Intelligence for Industries (AI4I 2020), pp. 69-74. IEEE.
2. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. In Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 4765-4774.
3. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), 56-67.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794.
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
6. UCI Machine Learning Repository. (2020). *AI4I 2020 Predictive Maintenance Dataset*. https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
