import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_raw_data

def run_full_eda(save_dir='artifacts/figures/'):
    """Runs full EDA and saves all figures."""
    os.makedirs(save_dir, exist_ok=True)
    df = load_raw_data()
    
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # 1. Target distribution bar chart
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Machine failure')
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 10, f'{height}\n({height/total:.1%})', ha="center")
    plt.title('Target Distribution (Machine failure)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'target_distribution.png'), dpi=150)
    plt.close()

    # 2. Numerical feature distributions (histograms)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'numerical_distributions.png'), dpi=150)
    plt.close()

    # 3. Boxplots for numerical features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplots.png'), dpi=150)
    plt.close()

    # 4. Correlation heatmap (only numerical features, NOT including target-leak columns)
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # 5. Failure rate by product type (bar chart)
    plt.figure(figsize=(8, 6))
    failure_rate = df.groupby('Type')['Machine failure'].mean().reset_index()
    failure_rate['Machine failure'] = failure_rate['Machine failure'] * 100
    sns.barplot(data=failure_rate, x='Type', y='Machine failure')
    plt.title('Failure Rate by Product Type')
    plt.ylabel('Failure Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'failure_by_type.png'), dpi=150)
    plt.close()

    # 6. Feature distributions grouped by failure status (overlapping histograms)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue='Machine failure', kde=True, ax=axes[i], common_norm=False, stat='density')
        axes[i].set_title(f'{col} by Failure Status')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_by_failure.png'), dpi=150)
    plt.close()

    # 7. Pairplot of important features colored by failure
    plt.figure(figsize=(12, 10))
    subset_cols = ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]', 'Machine failure']
    sns.pairplot(df[subset_cols], hue='Machine failure', corner=True)
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_overview.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    run_full_eda()
