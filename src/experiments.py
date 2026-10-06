"""
Research Experiments Module for X-Maintain.

Runs systematic research-style experiments:
Experiment 1: Baseline Logistic Regression (Class-weighted)
Experiment 2: Random Forest (Class-weighted)
Experiment 3: XGBoost (Default / Unweighted)
Experiment 4: Imbalance Strategies on XGBoost:
    - 4a. XGBoost with scale_pos_weight (Cost-Sensitive Weighting)
    - 4b. XGBoost with SMOTE (Synthetic Minority Over-sampling on training split only)
    - 4c. XGBoost with Threshold Tuning (Optimized for F1 / Recall balance)
Experiment 5: XAI validation and evaluation
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from src.preprocessing import RANDOM_STATE

def evaluate_predictions(y_true, y_pred, y_proba):
    """Calculates all key metrics for evaluation."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def run_all_experiments(
    processed_dir: str = 'data/processed',
    metrics_dir: str = 'artifacts/metrics',
    figures_dir: str = 'artifacts/figures'
):
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("X-Maintain: Running Comprehensive Research-Style Experiments")
    print("=" * 70)

    # Load preprocessed splits
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    experiment_results = {}

    # -------------------------------------------------------------
    # Experiment 1: Baseline Logistic Regression (Balanced)
    # -------------------------------------------------------------
    print("\n[Experiment 1] Logistic Regression (Balanced Weights)...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    experiment_results['Exp 1: Logistic Regression (Balanced)'] = evaluate_predictions(y_test, lr_pred, lr_proba)

    # -------------------------------------------------------------
    # Experiment 2: Random Forest (Balanced)
    # -------------------------------------------------------------
    print("[Experiment 2] Random Forest (Balanced Weights)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    experiment_results['Exp 2: Random Forest (Balanced)'] = evaluate_predictions(y_test, rf_pred, rf_proba)

    # -------------------------------------------------------------
    # Experiment 3: XGBoost (Unweighted / Standard)
    # -------------------------------------------------------------
    print("[Experiment 3] XGBoost (Unweighted / Standard)...")
    xgb_default = XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_default.fit(X_train, y_train)
    xgb_def_pred = xgb_default.predict(X_test)
    xgb_def_proba = xgb_default.predict_proba(X_test)[:, 1]
    experiment_results['Exp 3: XGBoost (Unweighted)'] = evaluate_predictions(y_test, xgb_def_pred, xgb_def_proba)

    # -------------------------------------------------------------
    # Experiment 4a: XGBoost with Cost-Sensitive Weighting
    # -------------------------------------------------------------
    print("[Experiment 4a] XGBoost with Cost-Sensitive Weighting (scale_pos_weight)...")
    xgb_weighted = XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    xgb_weighted.fit(X_train, y_train)
    xgb_w_pred = xgb_weighted.predict(X_test)
    xgb_w_proba = xgb_weighted.predict_proba(X_test)[:, 1]
    experiment_results['Exp 4a: XGBoost (Cost-Sensitive)'] = evaluate_predictions(y_test, xgb_w_pred, xgb_w_proba)

    # -------------------------------------------------------------
    # Experiment 4b: XGBoost with SMOTE (Applied ONLY on Train Data)
    # -------------------------------------------------------------
    print("[Experiment 4b] XGBoost with SMOTE (Train-split only)...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    xgb_smote = XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_smote.fit(X_train_smote, y_train_smote)
    xgb_sm_pred = xgb_smote.predict(X_test)
    xgb_sm_proba = xgb_smote.predict_proba(X_test)[:, 1]
    experiment_results['Exp 4b: XGBoost (SMOTE on Train)'] = evaluate_predictions(y_test, xgb_sm_pred, xgb_sm_proba)

    # -------------------------------------------------------------
    # Experiment 4c: XGBoost with Threshold Tuning (Tuned on Val Set)
    # -------------------------------------------------------------
    print("[Experiment 4c] XGBoost with Decision Threshold Tuning (Val-tuned)...")
    val_proba = xgb_weighted.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    
    # Avoid zero division
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    print(f"  Optimal decision threshold tuned on validation split: {optimal_threshold:.4f}")

    xgb_thresh_pred = (xgb_w_proba >= optimal_threshold).astype(int)
    exp4c_metrics = evaluate_predictions(y_test, xgb_thresh_pred, xgb_w_proba)
    exp4c_metrics['optimal_threshold'] = optimal_threshold
    experiment_results['Exp 4c: XGBoost (Threshold Tuned)'] = exp4c_metrics

    # -------------------------------------------------------------
    # Format and Save Results
    # -------------------------------------------------------------
    summary_rows = []
    for exp_name, metrics in experiment_results.items():
        summary_rows.append({
            'Experiment': exp_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall (Failure)': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Selection Score': (0.4 * metrics['recall'] + 0.3 * metrics['f1'] + 0.3 * metrics['roc_auc'])
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by='Selection Score', ascending=False, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    csv_path = os.path.join(metrics_dir, 'research_experiments_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved research experiment comparison table to: {csv_path}")

    json_path = os.path.join(metrics_dir, 'research_experiments.json')
    with open(json_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"Saved full experiment details to: {json_path}")

    # Plot comparison chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.2

    plt.bar(x - width, summary_df['Recall (Failure)'], width, label='Recall (Failure)', color='#e74c3c')
    plt.bar(x, summary_df['F1-Score'], width, label='F1-Score', color='#3498db')
    plt.bar(x + width, summary_df['Selection Score'], width, label='Selection Score', color='#2ecc71')

    plt.xlabel('Experiment Configuration', fontsize=11, fontweight='bold')
    plt.ylabel('Score', fontsize=11, fontweight='bold')
    plt.title('Research Experiments: Model & Imbalance Handling Comparison', fontsize=13, fontweight='bold')
    plt.xticks(x, [n.split(': ')[1] for n in summary_df['Experiment']], rotation=20, ha='right', fontsize=9)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    chart_path = os.path.join(figures_dir, 'imbalance_experiments_comparison.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to: {chart_path}")

    print("\n--- Research Experiment Summary ---")
    print(summary_df.to_string(index=False))
    print("=" * 70)

    return summary_df

if __name__ == '__main__':
    run_all_experiments()
