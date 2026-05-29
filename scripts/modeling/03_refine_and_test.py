"""
Phase 3: Refinement & Final Evaluation
- Compare SMOTE vs class_weight on top models (LR, RF)
- Final evaluation on held-out test set
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================
data = np.load('/home/claude/modeling/split_data.npz', allow_pickle=True)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
feature_names = list(data['feature_names'])
out_dir = '/home/claude/modeling'

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Train class dist: Control={sum(y_train==0)}, Dementia={sum(y_train==1)}")

# ============================================================
# 2. SMOTE ON TRAINING SET
# ============================================================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {X_train_sm.shape}")
print(f"  Control={sum(y_train_sm==0)}, Dementia={sum(y_train_sm==1)}")

# ============================================================
# 3. COMPARE: class_weight vs SMOTE
# ============================================================
print(f"\n{'='*75}")
print(f"SMOTE vs class_weight COMPARISON (Validation Set)")
print(f"{'='*75}")
print(f"{'Model':<30} {'Method':<15} {'F1':>6} {'AUC':>6} {'Acc':>6} {'Rec_C':>6} {'Rec_D':>6}")
print(f"{'-'*75}")

comparison_results = []

for name, ModelClass, kwargs_cw, kwargs_no in [
    ('Logistic Regression', LogisticRegression,
     dict(class_weight='balanced', max_iter=1000, random_state=42),
     dict(max_iter=1000, random_state=42)),
    ('SVM (RBF)', SVC,
     dict(class_weight='balanced', kernel='rbf', probability=True, random_state=42),
     dict(kernel='rbf', probability=True, random_state=42)),
    ('Random Forest', RandomForestClassifier,
     dict(class_weight='balanced', n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
     dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)),
]:
    # class_weight approach
    m_cw = ModelClass(**kwargs_cw)
    m_cw.fit(X_train, y_train)
    y_pred_cw = m_cw.predict(X_val)
    y_proba_cw = m_cw.predict_proba(X_val)[:, 1]
    f1_cw = f1_score(y_val, y_pred_cw, average='macro')
    auc_cw = roc_auc_score(y_val, y_proba_cw)
    acc_cw = accuracy_score(y_val, y_pred_cw)
    rec_c_cw = recall_score(y_val, y_pred_cw, pos_label=0)
    rec_d_cw = recall_score(y_val, y_pred_cw, pos_label=1)
    
    # SMOTE approach (no class_weight)
    m_sm = ModelClass(**kwargs_no)
    m_sm.fit(X_train_sm, y_train_sm)
    y_pred_sm = m_sm.predict(X_val)
    y_proba_sm = m_sm.predict_proba(X_val)[:, 1]
    f1_sm = f1_score(y_val, y_pred_sm, average='macro')
    auc_sm = roc_auc_score(y_val, y_proba_sm)
    acc_sm = accuracy_score(y_val, y_pred_sm)
    rec_c_sm = recall_score(y_val, y_pred_sm, pos_label=0)
    rec_d_sm = recall_score(y_val, y_pred_sm, pos_label=1)
    
    print(f"{name:<30} {'class_wt':<15} {f1_cw:>6.3f} {auc_cw:>6.3f} {acc_cw:>6.3f} {rec_c_cw:>6.3f} {rec_d_cw:>6.3f}")
    print(f"{'':<30} {'SMOTE':<15} {f1_sm:>6.3f} {auc_sm:>6.3f} {acc_sm:>6.3f} {rec_c_sm:>6.3f} {rec_d_sm:>6.3f}")
    
    comparison_results.append({
        'model': name, 'method': 'class_weight',
        'f1': f1_cw, 'auc': auc_cw, 'acc': acc_cw,
        'rec_control': rec_c_cw, 'rec_dementia': rec_d_cw
    })
    comparison_results.append({
        'model': name, 'method': 'SMOTE',
        'f1': f1_sm, 'auc': auc_sm, 'acc': acc_sm,
        'rec_control': rec_c_sm, 'rec_dementia': rec_d_sm
    })

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv(f'{out_dir}/smote_comparison.csv', index=False)

# Find best overall combo
best_row = comp_df.loc[comp_df['f1'].idxmax()]
print(f"\n>>> Best combo: {best_row['model']} + {best_row['method']} (F1={best_row['f1']:.3f})")

# ============================================================
# 4. FINAL TEST SET EVALUATION (Best 2 models)
# ============================================================
print(f"\n{'='*75}")
print(f"FINAL TEST SET EVALUATION")
print(f"{'='*75}")

# Train final models on train set with best method per model
final_models = {
    'Logistic Regression (class_wt)': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'SVM (class_wt)': SVC(
        class_weight='balanced', kernel='rbf', probability=True, random_state=42
    ),
    'Random Forest (class_wt)': RandomForestClassifier(
        class_weight='balanced', n_estimators=300, min_samples_leaf=5, 
        random_state=42, n_jobs=-1
    ),
    'XGBoost (scale_pos_wt)': XGBClassifier(
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric='logloss', verbosity=0
    ),
}

# Also add SMOTE versions of top models
final_models_smote = {}
for sname, ModelClass, kwargs in [
    ('Random Forest (SMOTE)', RandomForestClassifier,
     dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)),
]:
    final_models_smote[sname] = ModelClass(**kwargs)

test_results = {}

print(f"\n{'Model':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Spec':>6} {'Sens':>6}")
print(f"{'-'*75}")

for name, model in final_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # TN / (TN+FP)
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])   # TP / (TP+FN)
    
    test_results[name] = {
        'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec,
        'f1_macro': f1, 'roc_auc': auc, 'specificity': specificity,
        'sensitivity': sensitivity, 'confusion_matrix': cm,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    print(f"{name:<35} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f} {specificity:>6.3f} {sensitivity:>6.3f}")

for name, model in final_models_smote.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
    
    test_results[name] = {
        'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec,
        'f1_macro': f1, 'roc_auc': auc, 'specificity': specificity,
        'sensitivity': sensitivity, 'confusion_matrix': cm,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    print(f"{name:<35} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f} {specificity:>6.3f} {sensitivity:>6.3f}")

# ============================================================
# 5. TEST SET PLOTS
# ============================================================

# 5a. Test ROC curves
colors = ['#2E75B6', '#E67E22', '#27AE60', '#C0392B', '#8E44AD']
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, res) in enumerate(test_results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (Test Set)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/test_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 5b. Test confusion matrices
fig, axes = plt.subplots(1, len(test_results), figsize=(4*len(test_results), 3.5))
if len(test_results) == 1:
    axes = [axes]
for i, (name, res) in enumerate(test_results.items()):
    cm = res['confusion_matrix']
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_yticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    if i == 0:
        ax.set_ylabel('Actual', fontsize=10)
    short_name = name.replace(' (class_wt)', '\n(class_wt)').replace(' (scale_pos_wt)', '\n(scale_pos_wt)').replace(' (SMOTE)', '\n(SMOTE)')
    ax.set_title(short_name, fontsize=10, fontweight='bold')
    for r in range(2):
        for c in range(2):
            color = 'white' if cm[r,c] > cm.max()/2 else 'black'
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=color)
plt.suptitle('Confusion Matrices (Test Set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{out_dir}/test_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# 5c. Summary comparison bar chart
fig, ax = plt.subplots(figsize=(12, 5))
model_names = list(test_results.keys())
metrics = ['accuracy', 'f1_macro', 'roc_auc', 'sensitivity', 'specificity']
metric_labels = ['Accuracy', 'F1 (macro)', 'ROC-AUC', 'Sensitivity\n(Dementia Rec)', 'Specificity\n(Control Rec)']
x = np.arange(len(metrics))
width = 0.15

for i, name in enumerate(model_names):
    vals = [test_results[name][m] for m in metrics]
    short = name.split('(')[0].strip()
    method = name.split('(')[1].replace(')', '') if '(' in name else ''
    bars = ax.bar(x + i*width, vals, width, label=f"{short}\n({method})", 
                  color=colors[i % len(colors)], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Final Model Comparison (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylim(0, 1.12)
ax.legend(loc='lower right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved test plots to {out_dir}/")

# ============================================================
# 6. FINAL RANKING
# ============================================================
print(f"\n{'='*75}")
print("FINAL MODEL RANKING (Test Set)")
print(f"{'='*75}")
ranked = sorted(test_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
for i, (name, res) in enumerate(ranked, 1):
    print(f"  {i}. {name}")
    print(f"     F1={res['f1_macro']:.3f} | AUC={res['roc_auc']:.3f} | Acc={res['accuracy']:.3f} | Sens={res['sensitivity']:.3f} | Spec={res['specificity']:.3f}")

# Save test results
test_summary = {}
for name, res in test_results.items():
    test_summary[name] = {k: v for k, v in res.items() 
                          if k not in ('y_pred', 'y_proba', 'confusion_matrix')}
    test_summary[name]['confusion_matrix'] = res['confusion_matrix'].tolist()

import json
with open(f'{out_dir}/test_results.json', 'w') as f:
    json.dump(test_summary, f, indent=2)

print(f"\nAll results saved to {out_dir}/")
