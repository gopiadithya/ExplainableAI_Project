"""
Evaluation module for the X-Maintain project.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

RANDOM_STATE = 42

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """Evaluates a model and returns a dictionary of metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_pred': y_pred.tolist()
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        metrics['pr_auc'] = float(average_precision_score(y_test, y_proba))
        metrics['y_proba'] = y_proba.tolist()
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
        metrics['y_proba'] = None
        
    return metrics

def evaluate_all_models(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Evaluates all models and returns a comparison dict."""
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    return results

def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Returns a pandas DataFrame comparing all models."""
    data = []
    for name, metrics in results.items():
        data.append({
            'Model': name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1 Score': metrics.get('f1', np.nan),
            'ROC AUC': metrics.get('roc_auc', np.nan),
            'PR AUC': metrics.get('pr_auc', np.nan)
        })
    return pd.DataFrame(data).set_index('Model')

def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], save_dir: str = 'artifacts/figures/') -> None:
    """Plots confusion matrices for all models."""
    os.makedirs(save_dir, exist_ok=True)
    for name, metrics in results.items():
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cm_{name.replace(' ', '_').lower()}.png"), dpi=150)
        plt.close()

def plot_roc_curves(results: Dict[str, Dict[str, Any]], X_test: pd.DataFrame, y_test: pd.Series, save_dir: str = 'artifacts/figures/') -> None:
    """Plots ROC curves for all models on one plot."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for name, metrics in results.items():
        if metrics['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
            auc = metrics['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
    plt.close()

def plot_precision_recall_curves(results: Dict[str, Dict[str, Any]], X_test: pd.DataFrame, y_test: pd.Series, save_dir: str = 'artifacts/figures/') -> None:
    """Plots PR curves for all models."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for name, metrics in results.items():
        if metrics['y_proba'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, metrics['y_proba'])
            auc = metrics['pr_auc']
            plt.plot(recall, precision, label=f'{name} (PR AUC = {auc:.3f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=150)
    plt.close()

def select_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any, str]:
    """Selects the best model using weighted scoring.
    
    Prioritizes recall (important for predictive maintenance - missing failures is costly),
    but also considers F1 and ROC-AUC to ensure practical usability.
    
    Weights: Recall=0.4, F1=0.3, ROC-AUC=0.3
    """
    def get_score(name):
        recall = results[name].get('recall', 0)
        f1 = results[name].get('f1', 0)
        roc_auc = results[name].get('roc_auc', 0)
        return 0.4 * recall + 0.3 * f1 + 0.3 * roc_auc
    
    scores = {name: get_score(name) for name in results.keys()}
    best_model_name = max(scores, key=scores.get)
    
    reasoning = (
        f"Selected {best_model_name} based on weighted scoring "
        f"(Recall×0.4 + F1×0.3 + ROC-AUC×0.3 = {scores[best_model_name]:.4f}). "
        f"Metrics: Recall={results[best_model_name].get('recall', 0):.4f}, "
        f"F1={results[best_model_name].get('f1', 0):.4f}, "
        f"ROC-AUC={results[best_model_name].get('roc_auc', 0):.4f}. "
        f"In predictive maintenance, high recall is critical (missing failures is costly), "
        f"but F1 and ROC-AUC ensure the model is also practically usable."
    )
    
    # All model scores for comparison
    print("\n  Model Selection Scores:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = results[name]
        print(f"    {name}: Score={score:.4f} (Recall={r.get('recall',0):.4f}, F1={r.get('f1',0):.4f}, ROC-AUC={r.get('roc_auc',0):.4f})")
                 
    filepath = os.path.join('models/', f"{best_model_name.replace(' ', '_').lower()}.joblib")
    best_model = joblib.load(filepath) if os.path.exists(filepath) else None
                 
    return best_model_name, best_model, reasoning

def save_metrics(results: Dict[str, Dict[str, Any]], save_dir: str = 'artifacts/metrics/') -> None:
    """Saves metrics as JSON and the comparison table as CSV."""
    os.makedirs(save_dir, exist_ok=True)
    
    clean_results = {}
    for name, metrics in results.items():
        clean_results[name] = {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_proba', 'confusion_matrix']}
        
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(clean_results, f, indent=4)
        
    comp_df = create_comparison_table(results)
    comp_df.to_csv(os.path.join(save_dir, 'comparison_table.csv'))

if __name__ == '__main__':
    # 1. Loads saved models from models/
    print("Loading saved models...")
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    models = {}
    for name in model_names:
        filepath = os.path.join('models/', f"{name.replace(' ', '_').lower()}.joblib")
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
        else:
            print(f"Warning: Model {name} not found at {filepath}")
            
    # 2. Loads test data
    print("Loading test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # 3. Evaluates all models
    print("Evaluating all models...")
    results = evaluate_all_models(models, X_test, y_test)
    
    # 4. Creates and saves comparison table (done within save_metrics as per functions)
    print("Creating comparison table...")
    comp_df = create_comparison_table(results)
    
    # 5. Generates all plots
    print("Generating all plots...")
    plot_confusion_matrices(results)
    plot_roc_curves(results, X_test, y_test)
    plot_precision_recall_curves(results, X_test, y_test)
    
    # 6. Selects best model
    print("Selecting best model...")
    best_name, best_model, reasoning = select_best_model(results)
    
    # 7. Saves all metrics
    print("Saving all metrics...")
    save_metrics(results)
    
    # 8. Prints summary
    print("\n--- Evaluation Summary ---")
    print("Comparison Table:")
    print(comp_df.to_string())
    print(f"\nBest Model: {best_name}")
    print(f"Reasoning: {reasoning}")
    
    # Save best model as production model
    if best_model is not None:
        joblib.dump(best_model, 'models/best_model.joblib')
        joblib.dump(best_model, 'models/model.pkl')
        
        # Save best model info
        best_info = {
            "model_name": best_name,
            "reasoning": reasoning,
            "metrics": {k: v for k, v in results[best_name].items() if k not in ['y_pred', 'y_proba', 'confusion_matrix']}
        }
        with open('artifacts/metrics/best_model_info.json', 'w') as f:
            json.dump(best_info, f, indent=4)
        print(f"\nBest model saved as models/best_model.joblib and models/model.pkl")
    
    print("\nEvaluation completed successfully.")
