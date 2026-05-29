"""
Phase 2: Baseline Model Training & Evaluation
- Logistic Regression, SVM (RBF), Random Forest, XGBoost
- All with class_weight='balanced'
- Evaluate on validation set
- Feature importance analysis
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ============================================================
# 1. LOAD DATA
# ============================================================
data = np.load('/home/claude/modeling/split_data.npz', allow_pickle=True)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
feature_names = list(data['feature_names'])

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

out_dir = '/home/claude/modeling'

# ============================================================
# 2. DEFINE MODELS
# ============================================================
# Compute scale_pos_weight for XGBoost
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos  # < 1 since positive class is majority

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, C=1.0
    ),
    'SVM (RBF)': SVC(
        class_weight='balanced', kernel='rbf', probability=True, random_state=42, C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=300, max_depth=None,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        scale_pos_weight=1/scale_pos_weight,  # weight for minority class
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric='logloss', verbosity=0
    ),
}

# ============================================================
# 3. TRAIN AND EVALUATE
# ============================================================
results = {}

print(f"\n{'='*70}")
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"{'='*70}")

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='macro')
    rec = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    
    # Also compute per-class metrics
    prec_0 = precision_score(y_val, y_pred, pos_label=0)
    rec_0 = recall_score(y_val, y_pred, pos_label=0)
    f1_0 = f1_score(y_val, y_pred, pos_label=0)
    prec_1 = precision_score(y_val, y_pred, pos_label=1)
    rec_1 = recall_score(y_val, y_pred, pos_label=1)
    f1_1 = f1_score(y_val, y_pred, pos_label=1)
    
    results[name] = {
        'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec,
        'f1_macro': f1, 'roc_auc': auc, 'confusion_matrix': cm.tolist(),
        'y_pred': y_pred, 'y_proba': y_proba,
        'per_class': {
            'Control':  {'precision': prec_0, 'recall': rec_0, 'f1': f1_0},
            'Dementia': {'precision': prec_1, 'recall': rec_1, 'f1': f1_1},
        }
    }
    
    print(f"{name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")

# ============================================================
# 4. DETAILED CLASSIFICATION REPORTS
# ============================================================
print(f"\n{'='*70}")
print("DETAILED RESULTS PER MODEL")
print(f"{'='*70}")

for name, res in results.items():
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {res['accuracy']:.3f}")
    print(f"  ROC-AUC:   {res['roc_auc']:.3f}")
    print(f"  Macro F1:  {res['f1_macro']:.3f}")
    cm = np.array(res['confusion_matrix'])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>3}  FP={cm[0,1]:>3}   (Control:  {res['per_class']['Control']['precision']:.3f} prec, {res['per_class']['Control']['recall']:.3f} rec)")
    print(f"    FN={cm[1,0]:>3}  TP={cm[1,1]:>3}   (Dementia: {res['per_class']['Dementia']['precision']:.3f} prec, {res['per_class']['Dementia']['recall']:.3f} rec)")

# ============================================================
# 5. PLOTS
# ============================================================

# 5a. Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
metric_labels = ['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)', 'F1\n(macro)', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.18
colors = ['#2E75B6', '#E67E22', '#27AE60', '#C0392B']

for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=colors[i], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baseline Model Comparison (Validation Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylim(0, 1.08)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: baseline_comparison.png")

# 5b. Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for i, (name, res) in enumerate(results.items()):
    cm = np.array(res['confusion_matrix'])
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_yticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    if i == 0:
        ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    for r in range(2):
        for c in range(2):
            color = 'white' if cm[r,c] > cm.max()/2 else 'black'
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
plt.suptitle('Confusion Matrices (Validation Set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{out_dir}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: confusion_matrices.png")

# 5c. ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_val, res['y_proba'])
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (Validation Set)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: roc_curves.png")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

# Random Forest importances
rf_model = models['Random Forest']
rf_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# XGBoost importances
xgb_model = models['XGBoost']
xgb_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Logistic Regression coefficients (absolute)
lr_model = models['Logistic Regression']
lr_imp = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

# Plot feature importances
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# RF
ax = axes[0]
top_rf = rf_imp.head(12)
ax.barh(range(len(top_rf)), top_rf['importance'], color='#27AE60', edgecolor='white')
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Random Forest\nFeature Importance', fontsize=12, fontweight='bold')
ax.set_xlabel('Gini Importance')

# XGB
ax = axes[1]
top_xgb = xgb_imp.head(12)
ax.barh(range(len(top_xgb)), top_xgb['importance'], color='#C0392B', edgecolor='white')
ax.set_yticks(range(len(top_xgb)))
ax.set_yticklabels(top_xgb['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('XGBoost\nFeature Importance', fontsize=12, fontweight='bold')
ax.set_xlabel('Gain Importance')

# LR
ax = axes[2]
top_lr = lr_imp.head(12)
bar_colors = ['#2E75B6' if c > 0 else '#E67E22' for c in top_lr['coefficient']]
ax.barh(range(len(top_lr)), top_lr['abs_coefficient'], color=bar_colors, edgecolor='white')
ax.set_yticks(range(len(top_lr)))
ax.set_yticklabels(top_lr['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Logistic Regression\n|Coefficient| (blue=+, orange=\u2013)', fontsize=12, fontweight='bold')
ax.set_xlabel('|Coefficient|')

plt.suptitle('Feature Importance Across Models', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{out_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: feature_importance.png")

# Save importance tables
rf_imp.to_csv(f'{out_dir}/importance_rf.csv', index=False)
xgb_imp.to_csv(f'{out_dir}/importance_xgb.csv', index=False)
lr_imp.to_csv(f'{out_dir}/importance_lr.csv', index=False)

# ============================================================
# 7. SAVE RESULTS SUMMARY
# ============================================================
summary = {}
for name, res in results.items():
    summary[name] = {k: v for k, v in res.items() if k not in ('y_pred', 'y_proba')}
    
with open(f'{out_dir}/baseline_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print("PHASE 2 COMPLETE")
print(f"{'='*70}")

# Final ranking
print("\nModel Ranking by Macro F1:")
ranked = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
for i, (name, res) in enumerate(ranked, 1):
    print(f"  {i}. {name}: F1={res['f1_macro']:.3f}, AUC={res['roc_auc']:.3f}, Acc={res['accuracy']:.3f}")
