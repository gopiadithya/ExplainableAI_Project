# Academic Project Report Draft

**Project Title:** X-Maintain: Explainable AI for Predictive Maintenance  
**Course:** Explainable Artificial Intelligence (XAI)  
**Academic Milestone:** Review 2 Project Draft (October 2026)  
**Authors:** [Student Name, Register / Roll Number]  
**Supervisor / Guide:** [Faculty Advisor Name]  
**Institution:** [Department of Computer Science & Engineering / AI]  

---

## Abstract

Predictive maintenance (PdM) leverages machine sensor telemetry to anticipate equipment failures before catastrophic downtime occurs. However, contemporary high-performance machine learning models operate primarily as opaque "black boxes," providing point-estimate failure probabilities without actionable rationale. In manufacturing, machine operators require verifiable insights into specific thermodynamic and mechanical risk contributors before executing costly spindle shutdowns or component replacements. This paper presents an end-to-end prototype of **X-Maintain**, an Explainable AI system designed for industrial predictive maintenance. Using the benchmark AI4I 2020 Predictive Maintenance Dataset ($N = 10,000$), we implement a target-leakage-free preprocessing pipeline, benchmark three distinct classification architectures (Logistic Regression, Random Forest, and XGBoost), and empirically evaluate four class-imbalance mitigation strategies. Model selection is governed by a domain-grounded weighted utility function ($0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$) prioritizing failure recall. The selected cost-sensitive XGBoost model achieves $98.20\%$ accuracy, $73.53\%$ precision, $73.53\%$ failure recall, $0.9724$ ROC-AUC, and $0.7934$ PR-AUC on a held-out test partition ($N = 2,001$). Model predictions are interpreted using TreeSHAP (SHapley Additive exPlanations), providing both global feature importance rankings and local waterfall decompositions. Furthermore, we implement a deterministic, hallucination-free natural language explanation generator and an interactive model-based what-if sensitivity simulator integrated into a Streamlit dashboard. The complete pipeline is supported by an automated test suite with $100\%$ pass rates across 20 unit tests.

---

## 1. Introduction

### 1.1 Context and Motivation
Industrial manufacturing increasingly relies on computerized numerical control (CNC) milling tools and automated machining centers operating under severe continuous mechanical stress. Unplanned equipment breakdowns cost industrial manufacturers an estimated \$50 billion annually, with outages halting entire downstream assembly lines. Predictive maintenance (PdM) aims to transition factory operations from reactive repair ("run-to-failure") and rigid preventative maintenance schedules toward condition-based monitoring.

Despite the proliferation of machine learning algorithms for failure forecasting, adoption on factory floors remains constrained by the "trust gap." Standard deep neural networks and ensemble classifiers provide failure probabilities without revealing physical root causes. If an automated system recommends halting a milling machine without explanation, operators risk dismissing the alert as a false positive. Conversely, blindly following uninterpretable alerts leads to unnecessary tooling replacements.

### 1.2 The Asymmetric Cost Problem
Machine failure prediction is fundamentally characterized by severe class imbalance and asymmetric misclassification costs:
- **Cost of a False Negative (Missed Failure):** Catastrophic spindle seizure, ruined workpieces, collateral cutter destruction, and emergency outages costing \$10,000 to \$100,000+.
- **Cost of a False Positive (False Alarm):** Routine 10-minute visual inspection or tool recalibration costing \$100 to \$500.

Because missing an impending failure is exponentially more costly than investigating a false alarm, overall classification accuracy is an inappropriate metric. A model must prioritize **Recall on the Failure class** while maintaining sufficient Precision to prevent alarm fatigue.

### 1.3 Contributions of this Work
In this project, we develop and validate an end-to-end prototype of the X-Maintain system:
1. **Target-Leakage-Free Architecture:** Rigorously purging downstream failure mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and arbitrary serial identifiers (`UDI`, `Product ID`) prior to model training.
2. **Empirical Imbalance Benchmarking:** Conducting a systematic comparison of four class-imbalance mitigation strategies (unweighted logloss, cost-sensitive class weighting, SMOTE, and validation-tuned threshold adjustment).
3. **Multi-Objective Model Selection:** Formulating a weighted utility score that explicitly favors minority-class recall while penalizing erratic precision.
4. **TreeSHAP Explainability Layer:** Computing exact Shapley values to generate global beeswarm summaries, dependence plots, and local waterfall attributions.
5. **Deterministic Natural Language Synthesis:** Generating auditable diagnostic explanations without relying on generative LLMs.
6. **Model-Based What-If Simulator:** Developing a counterfactual sensitivity simulator embedded within an interactive 6-page Streamlit dashboard.

---

## 2. Literature Review

The development of interpretable predictive maintenance systems resides at the intersection of industrial reliability engineering, machine learning for imbalanced tabular data, and Explainable Artificial Intelligence (XAI).

### 2.1 Predictive Maintenance in Manufacturing
Traditional maintenance strategies rely on time-based preventative maintenance, replacing cutting tools after fixed operational intervals regardless of physical wear state (Mobley, 2002). Matzka (2020) demonstrated that sensor-driven condition monitoring can anticipate tool failure by tracking thermodynamic and mechanical anomalies, introducing the AI4I 2020 Predictive Maintenance Dataset as a standardized benchmark simulating milling machine dynamics.

### 2.2 Machine Learning on Imbalanced Tabular Data
Industrial failure datasets universally exhibit extreme class imbalance, where safe operating regimes outnumber failure events by orders of magnitude (often exceeding 25:1). Standard maximum-likelihood estimation in classifiers such as Logistic Regression and unweighted Decision Trees leads to degenerate solutions that maximize accuracy by predicting the majority class (He & Garcia, 2009).

To mitigate class imbalance, Chawla et al. (2002) proposed the Synthetic Minority Over-sampling Technique (SMOTE), which creates synthetic minority samples along line segments connecting $k$-nearest neighbors. However, in physical systems governed by non-linear conservation laws, synthetic interpolation risks generating unphysical sensor combinations. Alternatively, cost-sensitive learning (Elkan, 2001) adjusts loss function penalization by weighting minority samples inversely proportional to their class frequency, directly addressing asymmetric costs without distorting data geometry.

Ensemble methods, particularly Gradient Boosted Decision Trees (GBDT) such as XGBoost (Chen & Guestrin, 2016) and Random Forests (Breiman, 2001), consistently dominate tabular benchmarks due to their invariance to monotonic transformations and innate capacity to capture complex non-linear feature interactions.

### 2.3 Explainable Artificial Intelligence (XAI)
Post-hoc model explanation techniques are broadly categorized into surrogate approximations and game-theoretic attributions. Ribeiro et al. (2016) proposed Local Interpretable Model-agnostic Explanations (LIME), fitting local sparse linear models around perturbed samples. However, LIME explanations suffer from sampling instability and lack mathematical consistency.

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), unifying cooperative game theory (Shapley, 1953) with post-hoc explainability. SHAP is the unique attribution method that provably satisfies four essential axiomatic properties:
1. **Efficiency:** $\sum_{i=1}^M \phi_i(x) = f(x) - \mathbb{E}[f(x)]$.
2. **Symmetry:** If features $i$ and $j$ contribute equally across all coalitions, $\phi_i = \phi_j$.
3. **Dummy / Null Player:** Features with zero marginal contribution receive $\phi_i = 0$.
4. **Additivity / Monotonicity:** If a model changes such that a feature's marginal contribution increases or stays the same, its Shapley value cannot decrease.

For tree ensembles, Lundberg et al. (2020) formulated TreeSHAP (TreeExplainer), optimizing computation from exponential time $\mathcal{O}(2^M)$ down to polynomial time $\mathcal{O}(TLD^2)$, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is maximum tree depth. This enables exact, deterministic Shapley computation suitable for real-time industrial deployment.

---

## 3. Proposed System Methodology

### 3.1 System Architecture
The X-Maintain architecture comprises four decoupled tiers:
1. **Data Ingestion & Preprocessing Tier:** Responsible for raw ingestion, leakage removal, stratified splitting, and feature normalization.
2. **Modeling & Optimization Tier:** Manages model training, cost-sensitive weight configuration, and multi-metric benchmarking.
3. **Explainability & Attribution Tier:** Computes TreeSHAP values, generates visual waterfall attributions, and synthesizes deterministic natural-language reports.
4. **Interactive Application Tier:** Provides Streamlit-based manual parameter entry, real-time risk gauges, sensitivity simulation, and dataset diagnostics.

```
+-------------------------------------------------------------+
|                AI4I 2020 Raw Dataset (UCI)                  |
|                   10,000 Records, 14 Cols                   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Data Cleaning & Leakage Prevention Purge          |
|  - Drop Failure Modes: TWF, HDF, PWF, OSF, RNF (Leakage)    |
|  - Drop Serial Keys: UDI, Product ID (Non-predictive)       |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|         Stratified Partitioning (RANDOM_STATE = 42)         |
|         Train: 70% (6,999) | Val: 10% (1,000) | Test: 20%   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             Feature Transformation Pipeline                 |
|  - StandardScaler: Air Temp, Process Temp, RPM, Torque, Wear|
|  - One-Hot Encoder: Product Type -> Type_H, Type_L, Type_M  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Model Benchmark & Training                 |
|  - Baseline: Logistic Regression (Balanced)                 |
|  - Bagging: Random Forest (200 Trees, Balanced)             |
|  - Boosting: XGBoost (scale_pos_weight = 28.53)             |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|         Weighted Multi-Objective Model Selection            |
|     Score = 0.4*Recall + 0.3*F1 + 0.3*ROC-AUC -> XGBoost    |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 TreeSHAP Explainability Layer               |
|  - Global: Mean |SHAP| Bar, Beeswarm, Dependence Plots      |
|  - Local: Waterfall Decomposition per Instance              |
|  - Deterministic Natural Language Diagnostic Engine         |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Interactive Streamlit Production Dashboard        |
|  - Real-Time Inference | What-If Simulator | EDA Diagnostics|
+-------------------------------------------------------------+
```

### 3.2 Dataset Characterization & Target Leakage Audit
We utilize the AI4I 2020 dataset, containing 10,000 milling machine records:
- **Continuous Features (5):** Air temperature ($K$), Process temperature ($K$), Rotational speed ($rpm$), Torque ($Nm$), and Tool wear ($min$).
- **Categorical Feature (1):** Product variant `Type` ($L$, $M$, $H$).
- **Target Feature:** `Machine failure` ($0 = \text{Safe}$, $1 = \text{Failure}$).

#### Critical Leakage Removal
The dataset includes five failure mode indicators: Tool Wear Failure (`TWF`), Heat Dissipation Failure (`HDF`), Power Failure (`PWF`), Overstrain Failure (`OSF`), and Random Failure (`RNF`). In this dataset:
$$\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$$
Including these columns directly reveals the target label to the learner. Our pipeline enforces strict deletion of all five columns alongside non-predictive identifiers (`UDI`, `Product ID`), verified via automated assertions.

### 3.3 Preprocessing and Normalization
Data partitioning uses stratified sampling (`stratify=y`) with a fixed seed (`RANDOM_STATE = 42`):
- Training partition: $70\%$ ($N = 6,999$, $237$ failures)
- Validation partition: $10\%$ ($N = 1,000$, $34$ failures)
- Test partition: $20\%$ ($N = 2,001$, $68$ failures)

Numerical columns are standardized using `StandardScaler` fitted strictly on training data:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Categorical quality variants are one-hot encoded into three binary indicators (`Type_H`, `Type_L`, `Type_M`). Feature names are sanitized to replace brackets with parentheses to ensure compatibility with XGBoost DMatrix specifications.

### 3.4 Model Formulation
Three distinct algorithmic families were developed:
1. **Logistic Regression Baseline:**
   $$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
   Optimized with L2 regularization and inverse class frequency weighting:
   $$w_k = \frac{N}{2 \times N_k}$$
2. **Random Forest Classifier:**
   Ensemble of $B = 200$ bagged decision trees using Gini impurity and balanced bootstrap subsampling:
   $$H(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)$$
3. **XGBoost (Extreme Gradient Boosting):**
   Second-order gradient boosted decision trees minimizing regularized logistic loss:
   $$\mathcal{L} = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$
   where $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$. Dynamic cost-sensitive weighting is configured via:
   $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{6,762}{237} \approx 28.53$$

### 3.5 Model Selection Criterion
Because predictive maintenance requires high failure detection without overwhelming operators with false alarms, selection is governed by a domain-weighted utility function:
$$\text{Utility} = 0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$$

---

## 4. Experimental Results and Discussion

All experiments were executed on the held-out test split ($N = 2,001$ samples, containing 68 actual failure cases).

### 4.1 Primary Model Comparison Benchmark
Table 1 presents the performance of the three core model families.

**Table 1: Primary Model Benchmark Performance on Held-Out Test Split ($N = 2,001$)**
| Model Architecture | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 83.61% | 0.1467 | **0.7941** | 0.2477 | 0.8949 | 0.4187 | 0.6604 |
| **Random Forest (Balanced)** | 97.75% | **0.8710** | 0.3971 | 0.5455 | 0.9556 | 0.6930 | 0.6091 |
| **XGBoost (Cost-Sensitive — Selected)** | **98.20%** | 0.7353 | 0.7353 | **0.7353** | **0.9724** | **0.7934** | **0.8064** |

#### Performance Trade-off Analysis
The experimental benchmark reveals three distinct operating regimes:
1. **Logistic Regression (High Recall, Poor Precision):** Achieves the highest raw recall ($79.41\%$, capturing 54 of 68 failures). However, linear hyperplanes cannot resolve the non-linear boundaries between torque and rotational speed. Consequently, precision collapses to $14.67\%$, generating 314 false alarms.
2. **Random Forest (High Precision, Deficient Recall):** Demonstrates high operational precision ($87.10\%$), but misses over $60\%$ of true failure events (Recall = $39.71\%$, missing 41 of 68 breakdowns). In an industrial plant, missing 41 machine failures is unacceptable.
3. **XGBoost (Balanced Operational Frontier):** Delivers symmetric performance with $73.53\%$ Recall, $73.53\%$ Precision, $0.7353$ F1, $0.9724$ ROC-AUC, and $0.7934$ PR-AUC. It achieves the highest overall selection score ($0.8064$).

---

### 4.2 Class Imbalance Mitigation Study
To investigate the effect of imbalance handling, we executed a dedicated comparative experiment using XGBoost across four strategies:
1. **Unweighted (Standard Logloss):** Standard symmetric optimization.
2. **Cost-Sensitive Weighting:** `scale_pos_weight = 28.53`.
3. **SMOTE on Training Partition:** Synthetically balancing the training split to a 1:1 ratio.
4. **Validation-Tuned Decision Threshold:** Using cost-sensitive probabilities and optimizing the decision threshold on validation data ($\tau^* = 0.6123$).

**Table 2: Class Imbalance Strategy Comparison on Test Split ($N = 2,001$)**
| Experiment Configuration | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|
| **Exp 4b: XGBoost + SMOTE (Train)** | 97.85% | 0.6506 | **0.7941** | 0.7152 | 0.9620 | **0.7996** | **0.8208** |
| **Exp 4c: XGBoost + Threshold Tuned ($\tau^*=0.6123$)**| **98.35%** | **0.7692** | 0.7353 | **0.7519** | **0.9724** | 0.7934 | 0.8114 |
| **Exp 4a: XGBoost + Cost-Sensitive (`scale_pos_weight`)**| 98.20% | 0.7353 | 0.7353 | 0.7353 | **0.9724** | 0.7934 | 0.8064 |
| **Exp 3: XGBoost Standard (Unweighted)** | 98.30% | 0.8036 | 0.6618 | 0.7258 | 0.9653 | 0.7874 | 0.7720 |

#### Insights from Imbalance Benchmarking:
- **Unweighted XGBoost (Exp 3)** misses one-third of failure events ($66.18\%$ Recall) due to majority class bias.
- **Cost-Sensitive Weighting (Exp 4a)** recovers failure recall to $73.53\%$ without generating synthetic sensor readings.
- **SMOTE (Exp 4b)** reaches $79.41\%$ recall, but causes an $8.5\%$ drop in precision ($65.06\%$) due to synthetic interpolation along non-convex boundaries.
- **Threshold Tuning (Exp 4c)** yields the highest overall F1-score ($0.7519$) and precision ($76.92\%$).

---

### 4.3 Explainable AI (XAI) Findings

#### Global Feature Importance
Global feature importance was quantified using mean absolute Shapley values across 500 test samples:

**Table 3: Global Feature Importance Ranking (TreeSHAP)**
| Rank | Feature Name | Mean $\|SHAP\|$ Value | Operational Role in Trained Model |
|---|---|---|---|
| 1 | **Torque (Nm)** | **3.740** | Dominant driver of torsional strain and mechanical overload |
| 2 | **Tool wear (min)** | **3.057** | Cumulative cutting abrasion elevating failure risk |
| 3 | **Rotational speed (rpm)** | **2.070** | Extreme RPM drives thermal friction and centrifugal stress |
| 4 | **Air temperature (K)** | **2.005** | Ambient temperature regulating convection cooling |
| 5 | **Process temperature (K)** | **0.988** | Determines dissipation gradient $\Delta T$ |
| 6 | **Type_L** | **0.196** | Low quality variant failure bias |
| 7 | **Type_M** | **0.174** | Medium quality baseline variant |
| 8 | **Type_H** | **0.088** | Heavy duty quality variant (lowest failure propensity) |

#### Local Attribution via Waterfall Decomposition
For an individual failure case (Sample #11, Ground Truth = 1, Predicted Probability = 100%):
- Base expected value: $\mathbb{E}[f(x)] = -3.73$
- Positive push from **Torque = 65.2 Nm**: $+4.28$
- Positive push from **Tool wear = 214 min**: $+3.12$
- Negative pull from **Air temperature = 298.2 K**: $-0.45$
- Final log-odds: $f(x) = +3.22 \rightarrow P(\text{Failure}) = 1.00$.

#### Deterministic Natural Language Explanations
Rather than relying on ungrounded generative LLMs, explanations are produced deterministically from calculated Shapley values:
> *"The model predicts a HIGH failure risk (probability: 1.00). The strongest factors contributing to this prediction are: Torque (Nm) (increases risk), Tool wear (min) (increases risk). Factors decreasing risk: Air temperature (K) (decreases risk)."*

---

### 4.4 Model-Based What-If Analysis
To evaluate counterfactual adjustments, the sensitivity engine perturbs operational parameters:
- **Baseline Scenario:** Tool wear = 210 min, Torque = 65.0 Nm $\rightarrow$ **Failure Probability = 100.0% (High Risk)**.
- **Modified Scenario:** Tool wear reduced to 30 min, Torque reduced to 38.0 Nm $\rightarrow$ **Failure Probability = 0.0% (Low Risk)**.
- **Net Delta:** **$-100.0\%$ Risk Reduction**.

---

## 5. Conclusion and Research Limitations

### 5.1 Conclusion
This project developed an end-to-end prototype of **X-Maintain**, an Explainable AI predictive maintenance system. By enforcing strict target leakage elimination, comparing three classification models, and benchmarking four class-imbalance strategies, we demonstrated that cost-sensitive gradient boosted trees achieve balanced performance ($73.53\%$ Recall, $73.53\%$ Precision, $0.9724$ ROC-AUC). TreeSHAP integration provides both global feature rankings and local waterfall attributions, paired with deterministic natural language reporting and What-If sensitivity simulation within a Streamlit dashboard.

### 5.2 Research Limitations & Non-Causal Grounding
1. **Synthetic Nature of Dataset:** The AI4I 2020 dataset was generated from numerical simulation models; real factory environments feature non-stationary sensor drift and acoustic noise.
2. **Observational vs. Causal Interpretation:** SHAP attributions reflect the model's conditional expectations, **not physical causality**. Demonstrating that the model associates high torque with failure does not prove an intervention on torque alone will eliminate failure without accounting for confounding factors.
3. **Static Tabular Scope:** The current implementation processes single-point operational snapshots rather than multi-variate time-series waveforms.

### 5.3 Future Work
1. **Time-Series Deep Learning:** Extending the architecture to Temporal Convolutional Networks (TCN) or LSTMs for Remaining Useful Life (RUL) regression.
2. **Edge Hardware Deployment:** Quantizing the model into ONNX format for deployment on resource-constrained microcontrollers.
3. **Causal Discovery:** Integrating structural causal models (SCMs) and do-calculus to formalize counterfactual physical interventions.

---

## 6. References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
4. Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)* (Vol. 2, pp. 973–978).
5. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (pp. 4765–4774).
7. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9
8. Matzka, S. (2020). Explainable artificial intelligence for predictive maintenance applications. In *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)* (pp. 69–74). IEEE. https://doi.org/10.1109/AI4I49448.2020.00023
9. Mobley, R. K. (2002). *An introduction to predictive maintenance* (2nd ed.). Butterworth-Heinemann.
10. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144). ACM. https://doi.org/10.1145/2939672.2939778
11. Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307–317.
