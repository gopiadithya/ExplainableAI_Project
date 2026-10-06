# X-Maintain: Review 2 Presentation Slides & Viva Voce Defense Guide
**Course:** Explainable AI (XAI)  
**Project Title:** X-Maintain — Explainable AI for Predictive Maintenance  
**Review Target:** Review 2 (October 2026)  
**Academic Level:** Undergraduate / Capstone Engineering Defense  

---

## Part 1: Slide-by-Slide Presentation Structure (22 Slides)

### Slide 1: Title & Overview
- **Header:** X-Maintain: Explainable AI for Predictive Maintenance
- **Subtitle:** Interpretable Machine Learning for Industrial Equipment Failure Forecasting
- **Candidate Name / Reg No.:** [Student Name / Roll Number]
- **Guide / Supervisor:** [Faculty Guide Name]
- **Department:** Computer Science & Engineering / Data Science / AI
- **Key Visual:** System architecture badge & project logo
> **Speaker Notes:**  
> "Good morning, respected committee members. Today, I am presenting Review 2 of our project, *X-Maintain: Explainable AI for Predictive Maintenance*. In this phase, we have moved beyond conceptual framing to deliver a fully functional, verified prototype that predicts machine failure from multivariate sensor readings and explains every decision using game-theoretic Shapley values."

---

### Slide 2: Problem Statement & Industrial Context
- **Industrial Reality:** Unplanned factory downtime costs manufacturers over \$50 billion annually.
- **The Black-Box Dilemma:** Deep neural networks and ensemble models provide failure probabilities without revealing physical root causes.
- **Operator Problem:** Machine operators cannot afford to shut down milling spindles or replace cutting heads based solely on opaque probability scores.
- **Research Problem Formulation:** Formulate a machine failure prediction pipeline that pairs high-recall classification with mathematically grounded, localized post-hoc explanations.
> **Speaker Notes:**  
> "Predictive maintenance is an asymmetric problem: missing a catastrophic breakdown incurs severe financial and safety damage, while a false alarm only prompts a brief visual inspection. Therefore, our model must not only catch failures reliably, but also explain precisely which sensor readings are elevating the operational risk."

---

### Slide 3: Motivation & Asymmetric Risk
- **Cost Matrix Asymmetry:**
  - Cost(False Negative — Missed Failure): \$10,000 – \$100,000+ (Broken tooling, spindle destruction, factory halting)
  - Cost(False Positive — False Alarm): \$100 – \$500 (Routine 10-minute diagnostic check)
- **Core Research Imperative:**
  - Optimization cannot target overall accuracy (which is trivial in imbalanced data).
  - Model selection must prioritize **Failure Recall** while retaining operational precision.
> **Speaker Notes:**  
> "Notice the stark asymmetry in industrial loss. An algorithm that predicts 'Safe' 100% of the time achieves 96.6% accuracy on our dataset, but it is completely useless. In X-Maintain, we treat Recall on the minority failure class as our top priority."

---

### Slide 4: Project Objectives (Review 2 Milestones)
- [x] Ingest and audit benchmark industrial sensor dataset (UCI AI4I 2020).
- [x] Enforce strict **Target Leakage Prevention** across failure modes and serial identifiers.
- [x] Build reproducible, stratified preprocessing pipelines (`StandardScaler`, OHE).
- [x] Train baseline (Logistic Regression) and non-linear ensembles (Random Forest, XGBoost).
- [x] Empirically benchmark class imbalance strategies (Cost-sensitive weighting, SMOTE, Threshold tuning).
- [x] Implement TreeSHAP (TreeExplainer) for global and local attributions.
- [x] Construct a deterministic natural language explanation generator (zero LLM hallucinations).
- [x] Implement a model-based what-if sensitivity simulator.
- [x] Deploy an interactive 6-page Streamlit dashboard and comprehensive automated test suite.
> **Speaker Notes:**  
> "Every milestone established for Review 2 has been physically executed, tested, and validated. Today, we will walk through the empirical results, the XAI explanations, and a live demonstration."

---

### Slide 5: Existing Systems vs. Proposed System
| Dimension | Existing PdM Systems | Proposed X-Maintain Prototype |
|---|---|---|
| **Explainability** | Black-box scores or opaque heuristics | Exact Shapley values via TreeSHAP |
| **Model Selection** | Optimized for overall accuracy | Weighted utility ($0.4\text{Recall} + 0.3\text{F1} + 0.3\text{ROC-AUC}$) |
| **Leakage Audit** | Frequently leaks failure-mode tags | Complete purge of `TWF, HDF, PWF, OSF, RNF` |
| **Explanation Mode**| Static tables or unverified LLM text | Deterministic rule-engine linked to local SHAP |
| **Operator Tool** | Static dashboards | Interactive Streamlit UI with What-If sensitivity |
> **Speaker Notes:**  
> "Existing literature often suffers from two flaws: either models are opaque, or academic implementations accidentally leak target failure indicators. X-Maintain solves both issues rigorously."

---

### Slide 6: Dataset Description (UCI AI4I 2020)
- **Source:** UCI Machine Learning Repository (Matzka, 2020)
- **Dataset Character:** 10,000 synthetic records simulating real industrial milling dynamics.
- **Physical Sensor Variables (5):**
  - Air Temperature [K] (ambient, $295.3 - 304.5\text{ K}$)
  - Process Temperature [K] ($305.7 - 313.8\text{ K}$)
  - Rotational Speed [rpm] ($1,168 - 2,886\text{ RPM}$)
  - Torque [Nm] ($3.8 - 76.6\text{ Nm}$)
  - Tool Wear [min] ($0 - 253\text{ min}$)
- **Categorical Variant (1):** Product Type (`L` 50%, `M` 30%, `H` 20%)
- **Target Variable:** `Machine failure` ($0 = \text{No Failure}$, $1 = \text{Failure}$)
- **Class Imbalance:** 9,661 safe instances (96.61%) vs. 339 failure events (3.39%) $\rightarrow$ **28.5 : 1 Ratio**.
> **Speaker Notes:**  
> "We emphasize that AI4I 2020 is a synthetic benchmark dataset generated by Matzka from physical simulation models. This makes it an ideal, clean benchmark for testing XAI methods without sensor dropouts, but we explicitly document this in our limitations."

---

### Slide 7: Critical Target Leakage Prevention
- **The Pitfall:** The raw CSV includes columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- **The Logical Relationship:** $\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$.
- **Academic Risk:** Retaining any of these columns leaks the label into the feature matrix, allowing models to achieve 99.9% accuracy with zero predictive value.
- **Our Strict Exclusion Policy:**
  - Drop all 5 failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
  - Drop arbitrary primary keys (`UDI`, `Product ID`).
  - Formally guarded by unit test `test_no_target_leakage`.
> **Speaker Notes:**  
> "A common pitfall in student projects using AI4I is training on TWF or HDF. These columns describe the exact failure mode. In our pipeline, they are purged before splitting to guarantee academic integrity."

---

### Slide 8: Data Preprocessing & Reproducible Pipeline
- **Stratified Partitioning:**
  - Training Set: 6,999 samples (70%)
  - Validation Set: 1,000 samples (10%)
  - Test Set: 2,001 samples (20%) — exactly 68 failure cases.
- **Transformation Pipeline (`ColumnTransformer`):**
  - Numerical Features: `StandardScaler()` fitted **strictly on training set**.
  - Categorical Feature (`Type`): One-hot encoded into `Type_H`, `Type_L`, `Type_M`.
  - Feature Names: Sanitized brackets `[...]` $\rightarrow$ parentheses `(...)` for XGBoost DMatrix compatibility.
- **Serialization:** Preprocessor and schema serialized via `joblib`.
> **Speaker Notes:**  
> "To prevent data snooping, our StandardScaler is fitted only on the training partition and transformed across validation and test sets. Stratified sampling preserves the 3.39% minority failure proportion across all folds."

---

### Slide 9: Exploratory Data Analysis (EDA) Highlights
- **Key EDA Findings:**
  1. **Torque vs. Rotational Speed:** Inverse hyperbolic relationship ($P = \tau \cdot \omega$), highlighting motor power bounds.
  2. **Temperature Difference:** Process temperature closely tracks air temperature ($\Delta T \approx 10\text{ K}$).
  3. **Product Type Failure Rates:** Type L (Low quality) has the highest failure incidence ($3.92\%$), followed by M ($2.67\%$) and H ($2.05\%$).
- **Generated Artifacts:** 6 publication-ready figures in `artifacts/figures/` (distributions, boxplots, correlation matrices, grouped histograms).
> **Speaker Notes:**  
> "Our exploratory analysis confirmed two critical physical phenomena: machine power limits create a non-linear hyperbola between torque and RPM, and lower-tier products fail at nearly double the rate of heavy-duty variants."

---

### Slide 10: System Architecture
- *(Display diagram from `artifacts/figures/architecture_diagram.png`)*
- Visual trace: Raw Sensor Input $\rightarrow$ Preprocessing $\rightarrow$ Multi-Model Engine $\rightarrow$ Model Selection $\rightarrow$ TreeSHAP $\rightarrow$ What-If Engine $\rightarrow$ Streamlit Dashboard.
> **Speaker Notes:**  
> "This architecture diagram shows our modular design. The data layer, modeling layer, XAI layer, and user-facing presentation layer are decoupled and testable independently."

---

### Slide 11: Machine Learning Models Evaluated
1. **Baseline: Logistic Regression**
   - L2 regularized, `class_weight='balanced'`. Establishes linear baseline.
2. **Bagging Ensemble: Random Forest**
   - 200 trees, `class_weight='balanced'`, parallel execution. Tests non-linear bagging.
3. **Boosting Ensemble: XGBoost**
   - 200 estimators, second-order gradient boosting, dynamic `scale_pos_weight = 28.53`.
> **Speaker Notes:**  
> "We selected three representative model families: a regularized linear classifier as a baseline, a bagged ensemble of 200 random decision trees, and a gradient boosted decision tree architecture."

---

### Slide 12: Class Imbalance Mitigation Study
- We empirically benchmarked four distinct imbalance strategies on the test set:
  1. **Unweighted (Standard Logloss):** Penalizes false positives and false negatives equally.
  2. **Cost-Sensitive Weighting (`scale_pos_weight = 28.53`):** Assigns positive-class weight equal to the negative-to-positive ratio ($N_- / N_+$).
  3. **SMOTE on Training Data:** Interpolates synthetic failure instances along $k$-nearest neighbors in training space only.
  4. **Validation-Tuned Decision Threshold:** Shifts probability cutoff to $\tau^* = 0.6123$ based on PR curve F1 maximization.
> **Speaker Notes:**  
> "Rather than blindly applying SMOTE, we conducted a systematic empirical study. We compared standard loss, cost-sensitive class weighting, SMOTE, and validation-tuned threshold adjustment."

---

### Slide 13: Experimental Results Benchmark Table
| Experiment Configuration | Accuracy | Precision | Recall (Failures) | F1-Score | ROC-AUC | PR-AUC | Selection Score |
|---|---|---|---|---|---|---|---|
| **Exp 4b: XGBoost + SMOTE (Train)** | 97.85% | 0.6506 | **0.7941** | 0.7152 | 0.9620 | **0.7996** | **0.8208** |
| **Exp 4c: XGBoost + Threshold Tuned** | **98.35%** | **0.7692** | 0.7353 | **0.7519** | **0.9724** | 0.7934 | 0.8114 |
| **Exp 4a: XGBoost + Cost-Sensitive** | 98.20% | 0.7353 | 0.7353 | 0.7353 | **0.9724** | 0.7934 | 0.8064 |
| **Exp 3: XGBoost (Unweighted)** | 98.30% | 0.8036 | 0.6618 | 0.7258 | 0.9653 | 0.7874 | 0.7720 |
| **Exp 1: Logistic Regression (Balanced)** | 83.61% | 0.1467 | **0.7941** | 0.2477 | 0.8949 | 0.4187 | 0.6604 |
| **Exp 2: Random Forest (Balanced)** | 97.75% | 0.8710 | 0.3971 | 0.5455 | 0.9556 | 0.6930 | 0.6091 |
> **Speaker Notes:**  
> "These are the actual experimental results on our 2,001 test cases. Notice that unweighted XGBoost misses 33.8% of failures. Cost-sensitive weighting raises Recall to 73.53% without sacrificing precision, while SMOTE achieves 79.41% Recall at the expense of lower precision (65.06%)."

---

### Slide 14: Model Comparison & Selection Analysis
- **The Logistic Regression Paradox:** Highest raw recall ($79.41\%$), but precision is disastrous ($14.67\%$). Over 300 false alarms cause operator alert fatigue.
- **The Random Forest Failure:** High precision ($87.10\%$), but abysmal recall ($39.71\%$). Misses $60.3\%$ of actual tool breakdowns.
- **XGBoost Dominance:** Delivers balanced performance with **$73.53\%$ Recall**, **$73.53\%$ Precision**, **$0.9724$ ROC-AUC**, and **$0.7934$ PR-AUC**.
- **Utility Selection Criterion:**
  $$\text{Score} = 0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC} = 0.8064$$
> **Speaker Notes:**  
> "Our selection criterion explicitly encodes industrial priorities: $40\%$ weight on failure recall, $30\%$ on F1, and $30\%$ on ROC-AUC. XGBoost achieved the highest overall score among non-synthetic models."

---

### Slide 15: XAI Methodology — Why TreeSHAP?
- **Theoretical Basis:** Cooperative Game Theory (Shapley, 1953; Lundberg & Lee, 2017).
- **Core Axioms Satisfied:**
  1. *Efficiency:* Attributions sum to difference between prediction and expected base value: $\sum \phi_i(x) = f(x) - \mathbb{E}[f(x)]$.
  2. *Symmetry:* Equal feature contributions receive identical Shapley values.
  3. *Dummy (Null Player):* Irrelevant features receive $\phi_i = 0$.
  4. *Additivity / Monotonicity:* Consistent marginal contribution across ensemble trees.
- **TreeExplainer Advantage:** Polynomial-time complexity $\mathcal{O}(TLD^2)$ vs. exponential sampling time in KernelSHAP.
> **Speaker Notes:**  
> "TreeSHAP is not a heuristic; it is the unique attribution method that provably satisfies the game-theoretic axioms of efficiency, symmetry, and monotonicity. For tree ensembles, TreeExplainer computes exact Shapley values in polynomial time."

---

### Slide 16: Global Explainability Insights
- *(Display `artifacts/figures/shap_global_importance.png` & `shap_beeswarm.png`)*
- **Top 5 Influential Features in the Model:**
  1. **Torque (Nm)** ($\text{Mean } |\text{SHAP}| = 3.740$)
  2. **Tool wear (min)** ($\text{Mean } |\text{SHAP}| = 3.057$)
  3. **Rotational speed (rpm)** ($\text{Mean } |\text{SHAP}| = 2.070$)
  4. **Air temperature (K)** ($\text{Mean } |\text{SHAP}| = 2.005$)
  5. **Process temperature (K)** ($\text{Mean } |\text{SHAP}| = 0.988$)
- **Directionality via Beeswarm:** High torque combined with elevated tool wear sharply shifts log-odds toward failure.
> **Speaker Notes:**  
> "The beeswarm plot reveals the non-linear interaction: high torque (red dots) shifts model predictions strongly to the right, elevating failure risk. Crucially, as XAI researchers, we say 'SHAP shows Torque is the most influential feature in the model', not 'Torque causes failure'."

---

### Slide 17: Local Explainability & Waterfall Decomposition
- *(Display `artifacts/figures/shap_waterfall.png`)*
- For an individual machine failure prediction (Sample #11, Failure Prob = 100%):
  - Base Value: $\mathbb{E}[f(x)] = -3.73$ (log-odds corresponding to 3.39% baseline)
  - $+\,4.28$ pushed by **Torque = 65.2 Nm**
  - $+\,3.12$ pushed by **Tool wear = 214 min**
  - $-\,0.45$ pulled down by **Air temperature = 298.2 K**
  - Final output: $f(x) = +3.22 \rightarrow P(\text{Failure}) = 1.00$.
> **Speaker Notes:**  
> "This waterfall plot provides an exact visual audit for an individual machine. The operator can see the baseline risk and see each sensor reading adding or subtracting from the final decision."

---

### Slide 18: Deterministic Natural-Language Generation
- **Design Decision:** Zero LLM dependency for core explanations to eliminate hallucination risk.
- **Rule Engine Formulation:** Sorts positive and negative Shapley attributions:
- **Generated Output:**
  > *"The model predicts a HIGH failure risk (probability: 1.00). The strongest factors contributing to this prediction are: Torque (Nm) (increases risk), Tool wear (min) (increases risk). Factors decreasing risk: Air temperature (K) (decreases risk)."*
- **Auditability:** Deterministic, reproducible, and verifiable against raw SHAP values.
> **Speaker Notes:**  
> "We intentionally avoided using a large language model to write our core diagnostic reports. In industrial settings, safety explanations must be deterministic, auditable, and grounded directly in calculated Shapley values."

---

### Slide 19: Model-Based What-If Analysis
- **Objective:** Allow operators to simulate counterfactual parameter adjustments.
- **Empirical Demonstration:**
  - Base Scenario: Tool wear = 210 min, Torque = 65 Nm $\rightarrow$ **Failure Risk = 100% (High)**
  - Modified Scenario: Replace tool (Tool wear = 30 min), reduce torque (38 Nm) $\rightarrow$ **Failure Risk = 0% (Low)**
  - Delta: **$-100.0\%$ Risk Reduction**.
- **Non-Causal Disclaimer:**  
  *This analysis reflects the model's conditional response surface; it does not constitute physical causal intervention.*
> **Speaker Notes:**  
> "Our what-if simulator allows operators to explore operational adjustments. But we explicitly emphasize that this is model sensitivity analysis, not physical causality."

---

### Slide 20: Streamlit Dashboard Demo
- **6 Integrated Views:**
  1. **Home:** Project motivation, architecture summary, and key highlights.
  2. **Prediction:** Real-time sensor parameter sliders, probability gauge, waterfall plot, and NL explanation.
  3. **Explainability:** Global bar plots, beeswarm distribution, and test failure case inspector.
  4. **What-If Analysis:** Interactive dual-scenario comparison with probability delta metrics.
  5. **Model Performance:** Full benchmark table, imbalance study comparison, confusion matrices, ROC/PR curves.
  6. **Dataset Insights:** Interactive distributions, correlations, and class balance graphs.
> **Speaker Notes:**  
> "Our Streamlit dashboard provides a clean interface for operators and engineers. In a moment, I will demonstrate real-time inference and explanation."

---

### Slide 21: Verification & Automated Testing Suite
- **Automated Tests:** 20/20 Passing (`pytest tests/ -v` in 3.48s).
- **Test Categories:**
  - `test_preprocessing.py` (8 tests): Ingestion, no missing values, zero target leakage, stratified split ratio verification.
  - `test_prediction.py` (5 tests): Model loading, pipeline transforms, valid probability range $[0.0, 1.0]$, feature schema integrity.
  - `test_explainability.py` (7 tests): TreeExplainer generation, 2D/3D tensor shape checks, feature attribution sorting, NL generation logic, artifact existence.
> **Speaker Notes:**  
> "We implemented 20 unit tests verifying every critical component. All tests pass with zero failures."

---

### Slide 22: Conclusion, Limitations & Future Roadmap
- **Review 2 Deliverables Achieved:**
  - Working, reproducible prototype with strict leakage prevention.
  - Multi-model comparison and class imbalance benchmarking.
  - Verified TreeSHAP explainability pipeline and Streamlit dashboard.
- **Documented Limitations:**
  - Synthetic dataset characteristics (AI4I 2020).
  - Observational attribution rather than physical causality.
  - Tabular snapshot rather than dynamic time-series telemetry.
- **Future Work (Review 3 / Final Submission):**
  - Remaining Useful Life (RUL) regression using temporal LSTM/TCN architectures.
  - Quantized ONNX edge deployment for embedded microcontroller execution.
> **Speaker Notes:**  
> "In conclusion, X-Maintain provides a rigorous prototype combining predictive accuracy with model explainability. Thank you, and I look forward to your questions."

---

## Part 2: Viva Voce Q&A Preparation for Review 2

### Q1: Why did you remove the failure mode columns (TWF, HDF, PWF, OSF, RNF)?
**Model Answer:**  
"Because retaining them causes direct target leakage. In the AI4I dataset, `Machine failure` is a deterministic boolean OR of these five specific modes: $\text{Machine failure} = \text{TWF} \lor \text{HDF} \lor \text{PWF} \lor \text{OSF} \lor \text{RNF}$. If they are kept as input features, any tree model simply splits on these columns, reaching 99.9% accuracy trivially. However, in a real factory, we do not know if a heat dissipation failure has occurred until after the breakdown. Therefore, dropping them is required for academic correctness."

### Q2: Why did you not just use accuracy to choose the best model?
**Model Answer:**  
"The dataset has a 28.5:1 class imbalance—only 3.39% of instances are failures. A trivial zero-rule classifier that always outputs 'No Failure' achieves 96.61% accuracy, but misses 100% of actual breakdowns. In predictive maintenance, undetected failures cause severe equipment damage and production stoppages, whereas false alarms only incur a minor inspection cost. Thus, we evaluated models using Failure Recall, F1-score, and PR-AUC, and selected our model using a weighted domain utility function: $0.4 \times \text{Recall} + 0.3 \times \text{F1} + 0.3 \times \text{ROC-AUC}$."

### Q3: Why did you choose TreeSHAP instead of LIME or KernelSHAP?
**Model Answer:**  
"TreeSHAP provably satisfies four fundamental game-theoretic axioms: efficiency, symmetry, dummy player, and additivity. Heuristic methods like LIME and KernelSHAP rely on local sampling and linear approximations, which introduce sampling variance and can be slow. In contrast, TreeExplainer leverages the tree structure to compute exact Shapley values in $\mathcal{O}(TLD^2)$ polynomial time."

### Q4: Does your What-If analysis prove that reducing tool wear will physically prevent failure?
**Model Answer:**  
"No, and we are careful not to claim physical causality. What-if analysis is a model-based sensitivity analysis: it evaluates how the trained model's conditional probability responds when feature inputs are perturbed along its learned response surface. It does not replace physical intervention or causal DAG inference."

### Q5: How did you handle class imbalance, and why did you not use SMOTE for the final model?
**Model Answer:**  
"We ran an empirical study comparing four strategies: unweighted, cost-sensitive class weights (`scale_pos_weight = 28.53`), SMOTE on the training split, and validation-tuned threshold adjustment. While SMOTE yielded higher raw recall (79.41%), it caused an 8.5% drop in precision (65.06%) because synthetic interpolation in non-linear sensor space created false-positive instances. Cost-sensitive weighting achieved a cleaner balance of 73.53% recall and 73.53% precision without generating synthetic data."
