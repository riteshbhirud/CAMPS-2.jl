#!/usr/bin/env python3
"""
OFD Success Prediction - Machine Learning Training
===================================================

Trains ML models to predict OFD simulation success from pre-simulation
circuit features (no data leakage).

Features used:
- Circuit topology: n_qubits, n_t_gates, t_density
- Clifford depth statistics: avg, max, min, std
- Light cone widths: avg, max (w=4d relationship)
- Distribution metrics: spatial_uniformity, depth_variance
- Family: categorical variable

Target: OFD success rate >= 75% (binary: 1=success, 0=fail)
        Success = 3 or 4 out of 4 CAMPS runs succeeded

Usage:
    python train_ofd_model.py results/results_with_features_20260106.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(csv_path):
    """Load data and prepare features for ML training."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} circuits")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for any missing features
    feature_cols = [
        'n_qubits', 'n_t_gates', 
        'avg_clifford_depth', 'max_clifford_depth', 'min_clifford_depth', 'std_clifford_depth',
        'avg_light_cone_width', 'max_light_cone_width',
        'spatial_uniformity', 'depth_variance', 'n_gates'
    ]
    
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing feature columns: {missing}")
        sys.exit(1)
    
    print(f"\n✓ All {len(feature_cols)} feature columns present")
    
    # Compute t_density
    df['t_density'] = df['n_t_gates'] / df['n_qubits']
    print("✓ Computed t_density = n_t_gates / n_qubits")
    
    # Encode family as categorical
    le = LabelEncoder()
    df['family_encoded'] = le.fit_transform(df['family'])
    print(f"✓ Encoded {len(le.classes_)} families")
    
    # Print family distribution
    print("\nFamily distribution:")
    for i, family in enumerate(le.classes_):
        count = (df['family'] == family).sum()
        print(f"  {family:40s}: {count:3d} circuits")
    
    return df, le


def prepare_features_target(df):
    """Prepare feature matrix X and target vector y."""
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    # Feature columns (NO data leakage - all pre-simulation!)
    feature_cols = [
        'n_qubits',
        'n_t_gates',
        't_density',
        'avg_clifford_depth',
        'max_clifford_depth',
        'min_clifford_depth',
        'std_clifford_depth',
        'avg_light_cone_width',
        'max_light_cone_width',
        'spatial_uniformity',
        'depth_variance',
        'n_gates',
        'family_encoded'
    ]
    
    X = df[feature_cols].copy()
    
    # Convert ofd_rate to binary classification
    # Success = OFD rate >= 75% (3 or 4 out of 4 runs succeeded)
    y = (df['ofd_rate'] >= 0.75).astype(int)
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: OFD success rate >= 75% (binary)")
    print(f"\nTarget distribution:")
    print(f"  Success (1): {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Fail (0):    {len(y)-y.sum()} ({100*(1-y.mean()):.1f}%)")
    
    # Feature statistics
    print("\nFeature statistics:")
    print(X.describe().round(2))
    
    return X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple ML models and compare performance."""
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Scale features (important for SVM and LogReg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 70)
        
        # Use scaled data for SVM and LogReg, original for tree models
        if name in ['Logistic Regression', 'SVM (RBF)']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train.values, X_test.values
        
        # Train
        print("  Training...", end=" ")
        model.fit(X_tr, y_train)
        print("✓")
        
        # Predict
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            results[name]['auc'] = auc
        
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        if 'auc' in results[name]:
            print(f"  ROC AUC:   {auc:.3f}")
        
        trained_models[name] = model
    
    return results, trained_models, scaler


def analyze_best_model(model, X_train, X_test, y_train, y_test, feature_names, model_name="Random Forest"):
    """Detailed analysis of the best performing model."""
    print("\n" + "="*70)
    print(f"DETAILED ANALYSIS: {model_name}")
    print("="*70)
    
    # Predictions
    y_pred = model.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fail', 'Success'], zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Fail  Success")
    print(f"Actual Fail     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Success  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importance:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i, idx in enumerate(indices[:10]):  # Top 10
            print(f"  {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
        
        return importances
    
    return None


def create_visualizations(results, trained_models, X_test, y_test, feature_names, 
                         importances, output_dir):
    """Generate publication-quality visualizations."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[m][metric] for m in model_names]
        
        bars = ax.bar(range(len(model_names)), values, alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    fig_path = output_dir / 'model_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # 2. Feature importance (for Random Forest)
    if importances is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        indices = np.argsort(importances)[::-1][:15]  # Top 15
        
        ax.barh(range(len(indices)), importances[indices], alpha=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig_path = output_dir / 'feature_importance.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path}")
        plt.close()
    
    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, res in results.items():
        if 'y_pred_proba' in res and res['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            auc = res['auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - OFD Success Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'roc_curves.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # 4. Confusion matrix for best model (Random Forest)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    y_pred = results[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Success'],
                yticklabels=['Fail', 'Success'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = output_dir / 'confusion_matrix.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()


def save_results_summary(results, feature_names, importances, output_dir):
    """Save a text summary of results."""
    output_dir = Path(output_dir)
    summary_path = output_dir / 'results_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OFD SUCCESS PREDICTION - ML RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-"*70 + "\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall:    {res['recall']:.4f}\n")
            f.write(f"  F1 Score:  {res['f1']:.4f}\n")
            if 'auc' in res:
                f.write(f"  ROC AUC:   {res['auc']:.4f}\n")
        
        if importances is not None:
            f.write("\n\nFEATURE IMPORTANCE (Random Forest):\n")
            f.write("-"*70 + "\n")
            indices = np.argsort(importances)[::-1]
            for i, idx in enumerate(indices, 1):
                f.write(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}\n")
    
    print(f"✓ Saved: {summary_path}")


def main():
    """Main training pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python train_ofd_model.py <csv_file>")
        print("Example: python train_ofd_model.py results/results_with_features_20260106.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Load data
    df, label_encoder = load_and_prepare_data(csv_path)
    
    # Prepare features and target
    # Note: Using ofd_rate >= 0.75 as success threshold
    # This means 3 or 4 out of 4 runs must succeed
    X, y, feature_names = prepare_features_target(df)
    
    # Train/test split (80/20, stratified)
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set:   {len(X_train)} circuits ({100*len(X_train)/len(X):.1f}%)")
    print(f"Test set:       {len(X_test)} circuits ({100*len(X_test)/len(X):.1f}%)")
    print(f"Train success:  {y_train.sum()} / {len(y_train)} ({100*y_train.mean():.1f}%)")
    print(f"Test success:   {y_test.sum()} / {len(y_test)} ({100*y_test.mean():.1f}%)")
    
    # Train models
    results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Detailed analysis of best model (Random Forest)
    best_model = trained_models['Random Forest']
    importances = analyze_best_model(
        best_model, X_train, X_test, y_train, y_test, 
        feature_names, model_name="Random Forest"
    )
    
    # Create output directory
    output_dir = Path('ml_results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    create_visualizations(
        results, trained_models, X_test, y_test, 
        feature_names, importances, output_dir
    )
    
    # Save summary
    save_results_summary(results, feature_names, importances, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model: Random Forest")
    print(f"Test accuracy: {results['Random Forest']['accuracy']:.3f}")
    print(f"Test F1 score: {results['Random Forest']['f1']:.3f}")
    
    if importances is not None:
        top_3_idx = np.argsort(importances)[::-1][:3]
        print(f"\nTop 3 features:")
        for i, idx in enumerate(top_3_idx, 1):
            print(f"  {i}. {feature_names[idx]:30s} ({importances[idx]:.4f})")
    
    print(f"\nResults saved to: {output_dir}/")
    print("  - model_comparison.png")
    print("  - feature_importance.png")
    print("  - roc_curves.png")
    print("  - confusion_matrix.png")
    print("  - results_summary.txt")


if __name__ == '__main__':
    main()