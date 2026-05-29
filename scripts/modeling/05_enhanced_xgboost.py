"""
Script 05: Enhanced XGBoost Pipeline
=====================================
1. Add biomarker features from transcripts (filler, repetition, revision, fragment, unfilled_pause, utterance)
2. Add word_count and derived rate features (normalize counts by word_count)
3. Task-normalized z-scores for task-dependent features
4. Optuna hyperparameter tuning
5. Compare original (20 features) vs enhanced XGBoost

Paths: D:/FYP/ (ready to run, no edits needed)
Requirements: pip install optuna scikit-learn xgboost pandas numpy matplotlib
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_score, recall_score)
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    print("Optuna not found. Install with: pip install optuna")
    print("Falling back to manual grid search...")
    HAS_OPTUNA = False

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'D:/FYP/data/features/features_all_clean.csv'
TRANSCRIPTS_PATH = 'D:/FYP/data/transcripts/transcripts_clean.csv'
OUT_DIR = 'D:/FYP/results/Modeling Results'
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 1. LOAD AND MERGE DATA
# ============================================================
print(f"{'='*60}")
print("LOADING AND MERGING DATA")
print(f"{'='*60}")

features_df = pd.read_csv(FEATURES_PATH, encoding='latin-1')
transcripts_df = pd.read_csv(TRANSCRIPTS_PATH, encoding='latin-1')

print(f"Features: {features_df.shape}")
print(f"Transcripts: {transcripts_df.shape}")

# Create join keys
features_df['key'] = features_df['participant_id'].astype(str) + '_' + features_df['session'].astype(str) + '_' + features_df['task_folder']
transcripts_df['key'] = transcripts_df['participant_id'].astype(str) + '_' + transcripts_df['session'].astype(str) + '_' + transcripts_df['task_folder']

# Biomarker columns from transcripts
biomarker_cols = ['filler_count', 'unfilled_pause_count', 'repetition_count',
                  'revision_count', 'fragment_count', 'utterance_count']

df = pd.merge(
    features_df,
    transcripts_df[['key'] + biomarker_cols],
    on='key', how='inner'
)
print(f"Merged: {len(df)} records")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print(f"\n{'='*60}")
print("FEATURE ENGINEERING")
print(f"{'='*60}")

# Drop weak features from EDA (same as script 01)
df = df.drop(columns=['pause_rate', 'idea_density'], errors='ignore')

# --- Derived rate features (normalize counts by word_count) ---
df['filler_rate'] = df['filler_count'] / df['word_count'].clip(lower=1)
df['unfilled_pause_rate'] = df['unfilled_pause_count'] / df['word_count'].clip(lower=1)
df['repetition_rate_bio'] = df['repetition_count'] / df['word_count'].clip(lower=1)
df['revision_rate'] = df['revision_count'] / df['word_count'].clip(lower=1)
df['fragment_rate'] = df['fragment_count'] / df['word_count'].clip(lower=1)
df['utterance_rate'] = df['utterance_count'] / df['word_count'].clip(lower=1)
df['words_per_utterance'] = df['word_count'] / df['utterance_count'].clip(lower=1)

derived_cols = ['filler_rate', 'unfilled_pause_rate', 'repetition_rate_bio',
                'revision_rate', 'fragment_rate', 'utterance_rate', 'words_per_utterance']

# --- Task-normalized z-scores for task-dependent features ---
task_dependent = ['information_unit_coverage', 'story_recall_similarity',
                  'syntactic_complexity', 'semantic_coherence']

for feat in task_dependent:
    if feat in df.columns:
        col_name = f'{feat}_task_z'
        df[col_name] = df.groupby('task_folder')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        )

task_z_cols = [f'{f}_task_z' for f in task_dependent if f in df.columns]

print(f"Original features: 16 numeric + 4 task dummies = 20")
print(f"Added biomarker rates: {len(derived_cols)}")
print(f"Added task z-scores: {len(task_z_cols)}")
print(f"Added word_count: 1")

# ============================================================
# 3. DEFINE FEATURE SETS
# ============================================================
meta_cols = ['participant_id', 'session', 'task_folder', 'diagnosis', 'group',
             'label', 'word_count', 'key'] + biomarker_cols

# Original feature set (same as script 01)
original_numeric = ['syntactic_complexity', 'pronoun_noun_ratio', 'repetition_rate',
    'word_frequency_index', 'content_function_ratio', 'mean_pause_duration',
    'speech_time_ratio', 'pitch_range', 'pitch_variability', 'intensity_range',
    'jitter', 'shimmer', 'semantic_coherence', 'information_unit_coverage',
    'story_recall_similarity', 'global_semantic_drift']

# One-hot encode task
task_dummies = pd.get_dummies(df['task_folder'], prefix='task', dtype=int)
df = pd.concat([df, task_dummies], axis=1)
task_cols = list(task_dummies.columns)

original_features = original_numeric + task_cols

# Enhanced feature set
enhanced_numeric = original_numeric + derived_cols + task_z_cols + ['word_count']
enhanced_features = enhanced_numeric + task_cols

print(f"\nOriginal feature set: {len(original_features)} features")
print(f"Enhanced feature set: {len(enhanced_features)} features")

# ============================================================
# 4. PARTICIPANT-LEVEL SPLIT (same as script 01)
# ============================================================
print(f"\n{'='*60}")
print("PARTICIPANT-LEVEL SPLIT")
print(f"{'='*60}")

participant_df = df.groupby('participant_id').agg(
    group=('group', 'first'),
    label=('label', 'first'),
).reset_index()

train_pids, temp_pids = train_test_split(
    participant_df['participant_id'], test_size=0.20,
    random_state=SEED, stratify=participant_df['group']
)
temp_df = participant_df[participant_df['participant_id'].isin(temp_pids)]
val_pids, test_pids = train_test_split(
    temp_df['participant_id'], test_size=0.50,
    random_state=SEED, stratify=temp_df['group']
)

df['split'] = 'train'
df.loc[df['participant_id'].isin(val_pids), 'split'] = 'val'
df.loc[df['participant_id'].isin(test_pids), 'split'] = 'test'

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

# ============================================================
# 5. PREPARE FEATURE MATRICES
# ============================================================
def prepare_features(train_df, val_df, test_df, feature_cols, numeric_cols):
    X_train = train_df[feature_cols].values.astype(float)
    X_val = val_df[feature_cols].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)

    # Handle NaN/Inf
    for X in [X_train, X_val, X_test]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale numeric features only
    n_num = len(numeric_cols)
    scaler = StandardScaler()
    X_train[:, :n_num] = scaler.fit_transform(X_train[:, :n_num])
    X_val[:, :n_num] = scaler.transform(X_val[:, :n_num])
    X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

    return X_train, X_val, X_test

# Original features
X_train_orig, X_val_orig, X_test_orig = prepare_features(
    train_df, val_df, test_df, original_features, original_numeric)

# Enhanced features
X_train_enh, X_val_enh, X_test_enh = prepare_features(
    train_df, val_df, test_df, enhanced_features, enhanced_numeric)

print(f"Original X_train: {X_train_orig.shape}")
print(f"Enhanced X_train: {X_train_enh.shape}")

# ============================================================
# 6. BASELINE: ORIGINAL XGBoost (same params as script 02)
# ============================================================
print(f"\n{'='*60}")
print("BASELINE: ORIGINAL XGBOOST (20 features)")
print(f"{'='*60}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

baseline_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=1/scale_pos_weight,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
baseline_model.fit(X_train_orig, y_train)
baseline_preds = baseline_model.predict(X_test_orig)
baseline_probs = baseline_model.predict_proba(X_test_orig)[:, 1]

baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, average='macro')
baseline_auc = roc_auc_score(y_test, baseline_probs)

print(f"Acc={baseline_acc:.3f} | F1={baseline_f1:.3f} | AUC={baseline_auc:.3f}")

# ============================================================
# 7. ENHANCED: XGBoost with new features (default params)
# ============================================================
print(f"\n{'='*60}")
print("ENHANCED XGBOOST (default params, {0} features)".format(len(enhanced_features)))
print(f"{'='*60}")

enhanced_default = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=1/scale_pos_weight,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
enhanced_default.fit(X_train_enh, y_train)
enh_def_preds = enhanced_default.predict(X_test_enh)
enh_def_probs = enhanced_default.predict_proba(X_test_enh)[:, 1]

enh_def_acc = accuracy_score(y_test, enh_def_preds)
enh_def_f1 = f1_score(y_test, enh_def_preds, average='macro')
enh_def_auc = roc_auc_score(y_test, enh_def_probs)

print(f"Acc={enh_def_acc:.3f} | F1={enh_def_f1:.3f} | AUC={enh_def_auc:.3f}")

# ============================================================
# 8. OPTUNA HYPERPARAMETER TUNING
# ============================================================
print(f"\n{'='*60}")
print("HYPERPARAMETER TUNING (enhanced features)")
print(f"{'='*60}")

# Combine train + val for tuning, use cross-validation
X_trainval = np.vstack([X_train_enh, X_val_enh])
y_trainval = np.concatenate([y_train, y_val])

# Get participant IDs for proper CV splits
trainval_pids = pd.concat([train_df[['participant_id']], val_df[['participant_id']]]).values.ravel()
trainval_groups = pd.concat([train_df[['group']], val_df[['group']]]).values.ravel()

if HAS_OPTUNA:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5.0),
            'scale_pos_weight': 1/scale_pos_weight,
            'random_state': SEED,
            'eval_metric': 'logloss',
            'verbosity': 0,
        }

        # 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for fold_train_idx, fold_val_idx in cv.split(X_trainval, y_trainval):
            X_f_train, X_f_val = X_trainval[fold_train_idx], X_trainval[fold_val_idx]
            y_f_train, y_f_val = y_trainval[fold_train_idx], y_trainval[fold_val_idx]

            model = XGBClassifier(**params)
            model.fit(X_f_train, y_f_train)
            preds = model.predict(X_f_val)
            scores.append(f1_score(y_f_val, preds, average='macro'))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best_params = study.best_params
    best_cv_f1 = study.best_value
    print(f"\nBest CV F1: {best_cv_f1:.4f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

else:
    # Manual grid search fallback
    print("Running manual grid search (fewer combinations)...")
    best_cv_f1 = 0
    best_params = {}

    param_grid = [
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 300, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.7},
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 600, 'subsample': 0.7, 'colsample_bytree': 0.8},
        {'max_depth': 7, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.6},
        {'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': 400, 'subsample': 0.85, 'colsample_bytree': 0.75},
        {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 700, 'subsample': 0.75, 'colsample_bytree': 0.7},
        {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.9, 'colsample_bytree': 0.8},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, params in enumerate(param_grid):
        scores = []
        for fold_train_idx, fold_val_idx in cv.split(X_trainval, y_trainval):
            X_f_train, X_f_val = X_trainval[fold_train_idx], X_trainval[fold_val_idx]
            y_f_train, y_f_val = y_trainval[fold_train_idx], y_trainval[fold_val_idx]
            model = XGBClassifier(**params, scale_pos_weight=1/scale_pos_weight,
                                  random_state=SEED, eval_metric='logloss', verbosity=0)
            model.fit(X_f_train, y_f_train)
            preds = model.predict(X_f_val)
            scores.append(f1_score(y_f_val, preds, average='macro'))

        mean_f1 = np.mean(scores)
        print(f"  Config {i+1}/{len(param_grid)}: F1={mean_f1:.4f} (depth={params['max_depth']}, lr={params['learning_rate']}, n={params['n_estimators']})")
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_params = params

    print(f"\nBest CV F1: {best_cv_f1:.4f}")
    print(f"Best params: {best_params}")

# ============================================================
# 9. TRAIN FINAL TUNED MODEL ON TRAIN+VAL, EVALUATE ON TEST
# ============================================================
print(f"\n{'='*60}")
print("FINAL TUNED MODEL (test set evaluation)")
print(f"{'='*60}")

final_params = {**best_params, 'scale_pos_weight': 1/scale_pos_weight,
                'random_state': SEED, 'eval_metric': 'logloss', 'verbosity': 0}

tuned_model = XGBClassifier(**final_params)
tuned_model.fit(X_trainval, y_trainval)

tuned_preds = tuned_model.predict(X_test_enh)
tuned_probs = tuned_model.predict_proba(X_test_enh)[:, 1]

tuned_acc = accuracy_score(y_test, tuned_preds)
tuned_f1 = f1_score(y_test, tuned_preds, average='macro')
tuned_auc = roc_auc_score(y_test, tuned_probs)
tuned_prec = precision_score(y_test, tuned_preds, average='macro')
tuned_rec = recall_score(y_test, tuned_preds, average='macro')
tuned_cm = confusion_matrix(y_test, tuned_preds)
tuned_spec = tuned_cm[0,0] / (tuned_cm[0,0] + tuned_cm[0,1])
tuned_sens = tuned_cm[1,1] / (tuned_cm[1,0] + tuned_cm[1,1])

print(f"Accuracy:    {tuned_acc:.3f}")
print(f"F1 (macro):  {tuned_f1:.3f}")
print(f"ROC-AUC:     {tuned_auc:.3f}")
print(f"Precision:   {tuned_prec:.3f}")
print(f"Recall:      {tuned_rec:.3f}")
print(f"Sensitivity: {tuned_sens:.3f}")
print(f"Specificity: {tuned_spec:.3f}")
print(f"Confusion Matrix:")
print(f"  TN={tuned_cm[0,0]}  FP={tuned_cm[0,1]}")
print(f"  FN={tuned_cm[1,0]}  TP={tuned_cm[1,1]}")

# ============================================================
# 10. FEATURE IMPORTANCE (tuned model)
# ============================================================
importance_df = pd.DataFrame({
    'feature': enhanced_features,
    'importance': tuned_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features (tuned model):")
for i, row in importance_df.head(15).iterrows():
    marker = " ★ NEW" if row['feature'] in derived_cols + task_z_cols + ['word_count'] else ""
    print(f"  {row['feature']:<35} {row['importance']:.4f}{marker}")

importance_df.to_csv(f'{OUT_DIR}/importance_enhanced_xgb.csv', index=False)

# ============================================================
# 11. COMPARISON TABLE
# ============================================================
print(f"\n{'='*60}")
print("FINAL COMPARISON (TEST SET)")
print(f"{'='*60}")
print(f"{'Model':<45} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print(f"{'-'*65}")
print(f"{'XGBoost original (20 feat, default params)':<45} {baseline_acc:>6.3f} {baseline_f1:>6.3f} {baseline_auc:>6.3f}")
print(f"{'XGBoost enhanced ('+str(len(enhanced_features))+' feat, default params)':<45} {enh_def_acc:>6.3f} {enh_def_f1:>6.3f} {enh_def_auc:>6.3f}")
print(f"{'XGBoost enhanced + tuned':<45} {tuned_acc:>6.3f} {tuned_f1:>6.3f} {tuned_auc:>6.3f}")

# ============================================================
# 12. PLOTS
# ============================================================
colors = ['#C0392B', '#27AE60', '#2E75B6']

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_names = ['XGBoost\noriginal\n(20 feat)', f'XGBoost\nenhanced\n({len(enhanced_features)} feat)',
               'XGBoost\nenhanced\n+ tuned']
metrics_vals = {
    'Accuracy': [baseline_acc, enh_def_acc, tuned_acc],
    'F1 (macro)': [baseline_f1, enh_def_f1, tuned_f1],
    'ROC-AUC': [baseline_auc, enh_def_auc, tuned_auc],
}
x = np.arange(len(model_names))
width = 0.25
for i, (metric, vals) in enumerate(metrics_vals.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric, color=colors[i], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('XGBoost Enhancement Comparison (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.08)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/xgboost_enhancement_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs, color in [
    ('Original (20 feat)', baseline_probs, colors[0]),
    (f'Enhanced ({len(enhanced_features)} feat)', enh_def_probs, colors[1]),
    ('Enhanced + tuned', tuned_probs, colors[2]),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - XGBoost Enhancement (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/xgboost_enhancement_roc.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance plot (top 20)
fig, ax = plt.subplots(figsize=(10, 7))
top20 = importance_df.head(20)
bar_colors = ['#E67E22' if f in derived_cols + task_z_cols + ['word_count'] else '#2E75B6'
              for f in top20['feature']]
ax.barh(range(len(top20)), top20['importance'], color=bar_colors, edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Feature Importance - Enhanced XGBoost (blue=original, orange=new)', fontsize=12, fontweight='bold')
ax.set_xlabel('Gain Importance')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/xgboost_enhanced_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, preds_arr) in zip(axes, [
    ('Original', baseline_preds), (f'Enhanced', enh_def_preds), ('Enhanced+Tuned', tuned_preds)
]):
    cm = confusion_matrix(y_test, preds_arr)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_yticklabels(['Control', 'Dementia'], fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_title(name, fontsize=11, fontweight='bold')
    if ax == axes[0]: ax.set_ylabel('Actual')
    for r in range(2):
        for c in range(2):
            color = 'white' if cm[r,c] > cm.max()/2 else 'black'
            ax.text(c, r, str(cm[r,c]), ha='center', va='center', fontsize=16, fontweight='bold', color=color)
plt.suptitle('Confusion Matrices - XGBoost Enhancement (Test Set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/xgboost_enhanced_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 13. SAVE RESULTS
# ============================================================
results = {
    'original_xgboost': {
        'n_features': len(original_features),
        'accuracy': baseline_acc, 'f1_macro': baseline_f1, 'roc_auc': baseline_auc
    },
    'enhanced_xgboost_default': {
        'n_features': len(enhanced_features),
        'accuracy': enh_def_acc, 'f1_macro': enh_def_f1, 'roc_auc': enh_def_auc
    },
    'enhanced_xgboost_tuned': {
        'n_features': len(enhanced_features),
        'accuracy': tuned_acc, 'f1_macro': tuned_f1, 'roc_auc': tuned_auc,
        'precision_macro': tuned_prec, 'recall_macro': tuned_rec,
        'sensitivity': tuned_sens, 'specificity': tuned_spec,
        'confusion_matrix': tuned_cm.tolist(),
        'best_params': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                        for k, v in best_params.items()},
        'best_cv_f1': best_cv_f1,
    },
    'enhanced_features_list': enhanced_features,
    'new_features_added': derived_cols + task_z_cols + ['word_count'],
}

with open(f'{OUT_DIR}/enhanced_xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {OUT_DIR}/:")
print(f"  xgboost_enhancement_comparison.png")
print(f"  xgboost_enhancement_roc.png")
print(f"  xgboost_enhanced_importance.png")
print(f"  xgboost_enhanced_confusion.png")
print(f"  importance_enhanced_xgb.csv")
print(f"  enhanced_xgboost_results.json")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
