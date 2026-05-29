"""
Script 06: Stacking Ensemble + BERT Integration
=================================================
1. Load 32 enhanced features (same split as script 05)
2. Level 1: 4 base classifiers produce out-of-fold probabilities via 5-fold CV
3. Add BERT P(AD) as 5th meta-feature (from script 04 results)
4. Level 2: Meta-XGBoost trained on 5 probability features
5. Optuna tuning on the meta-learner
6. Final test set evaluation + comparison

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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_score, recall_score)
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'D:/FYP/data/features/features_all_clean.csv'
TRANSCRIPTS_PATH = 'D:/FYP/data/transcripts/transcripts_clean.csv'
BERT_RESULTS_PATH = 'D:/FYP/results/Modeling Results/multimodal_results.json'
OUT_DIR = 'D:/FYP/results/Modeling Results'
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 1. LOAD AND PREPARE DATA (same as script 05)
# ============================================================
print(f"{'='*60}")
print("LOADING DATA")
print(f"{'='*60}")

features_df = pd.read_csv(FEATURES_PATH, encoding='latin-1')
transcripts_df = pd.read_csv(TRANSCRIPTS_PATH, encoding='latin-1')

features_df['key'] = features_df['participant_id'].astype(str) + '_' + features_df['session'].astype(str) + '_' + features_df['task_folder']
transcripts_df['key'] = transcripts_df['participant_id'].astype(str) + '_' + transcripts_df['session'].astype(str) + '_' + transcripts_df['task_folder']

biomarker_cols = ['filler_count', 'unfilled_pause_count', 'repetition_count',
                  'revision_count', 'fragment_count', 'utterance_count']

df = pd.merge(features_df, transcripts_df[['key'] + biomarker_cols], on='key', how='inner')
df = df.drop(columns=['pause_rate', 'idea_density'], errors='ignore')

# Derived features
df['filler_rate'] = df['filler_count'] / df['word_count'].clip(lower=1)
df['unfilled_pause_rate'] = df['unfilled_pause_count'] / df['word_count'].clip(lower=1)
df['repetition_rate_bio'] = df['repetition_count'] / df['word_count'].clip(lower=1)
df['revision_rate'] = df['revision_count'] / df['word_count'].clip(lower=1)
df['fragment_rate'] = df['fragment_count'] / df['word_count'].clip(lower=1)
df['utterance_rate'] = df['utterance_count'] / df['word_count'].clip(lower=1)
df['words_per_utterance'] = df['word_count'] / df['utterance_count'].clip(lower=1)

derived_cols = ['filler_rate', 'unfilled_pause_rate', 'repetition_rate_bio',
                'revision_rate', 'fragment_rate', 'utterance_rate', 'words_per_utterance']

# Task z-scores
task_dependent = ['information_unit_coverage', 'story_recall_similarity',
                  'syntactic_complexity', 'semantic_coherence']
for feat in task_dependent:
    if feat in df.columns:
        df[f'{feat}_task_z'] = df.groupby('task_folder')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        )

task_z_cols = [f'{f}_task_z' for f in task_dependent if f in df.columns]

# Task dummies
task_dummies = pd.get_dummies(df['task_folder'], prefix='task', dtype=int)
df = pd.concat([df, task_dummies], axis=1)
task_cols = list(task_dummies.columns)

# Feature set
original_numeric = ['syntactic_complexity', 'pronoun_noun_ratio', 'repetition_rate',
    'word_frequency_index', 'content_function_ratio', 'mean_pause_duration',
    'speech_time_ratio', 'pitch_range', 'pitch_variability', 'intensity_range',
    'jitter', 'shimmer', 'semantic_coherence', 'information_unit_coverage',
    'story_recall_similarity', 'global_semantic_drift']
enhanced_numeric = original_numeric + derived_cols + task_z_cols + ['word_count']
enhanced_features = enhanced_numeric + task_cols

# Split
from sklearn.model_selection import train_test_split
participant_df = df.groupby('participant_id').agg(group=('group', 'first'), label=('label', 'first')).reset_index()
train_pids, temp_pids = train_test_split(participant_df['participant_id'], test_size=0.20, random_state=SEED, stratify=participant_df['group'])
temp_df_p = participant_df[participant_df['participant_id'].isin(temp_pids)]
val_pids, test_pids = train_test_split(temp_df_p['participant_id'], test_size=0.50, random_state=SEED, stratify=temp_df_p['group'])

df['split'] = 'train'
df.loc[df['participant_id'].isin(val_pids), 'split'] = 'val'
df.loc[df['participant_id'].isin(test_pids), 'split'] = 'test'

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

# Prepare arrays
def get_arrays(train_df, val_df, test_df, feature_cols, numeric_cols):
    X_tr = train_df[feature_cols].values.astype(float)
    X_v = val_df[feature_cols].values.astype(float)
    X_te = test_df[feature_cols].values.astype(float)
    for X in [X_tr, X_v, X_te]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(numeric_cols)
    sc = StandardScaler()
    X_tr[:, :n] = sc.fit_transform(X_tr[:, :n])
    X_v[:, :n] = sc.transform(X_v[:, :n])
    X_te[:, :n] = sc.transform(X_te[:, :n])
    return X_tr, X_v, X_te

X_train, X_val, X_test = get_arrays(train_df, val_df, test_df, enhanced_features, enhanced_numeric)
y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

# Combine train+val for stacking
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

print(f"Train+Val: {X_trainval.shape[0]} | Test: {X_test.shape[0]}")
print(f"Features: {X_trainval.shape[1]}")

# ============================================================
# 2. LEVEL 1: OUT-OF-FOLD PREDICTIONS
# ============================================================
print(f"\n{'='*60}")
print("LEVEL 1: BASE CLASSIFIERS (5-fold CV)")
print(f"{'='*60}")

scale_pw = (y_trainval == 0).sum() / (y_trainval == 1).sum()

base_models = {
    'XGBoost': XGBClassifier(
        n_estimators=483, max_depth=7, learning_rate=0.288,
        min_child_weight=2, subsample=0.802, colsample_bytree=0.947,
        reg_alpha=0.136, reg_lambda=0.076, gamma=0.09,
        scale_pos_weight=1/scale_pw, random_state=SEED, eval_metric='logloss', verbosity=0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, class_weight='balanced', min_samples_leaf=5,
        random_state=SEED, n_jobs=-1
    ),
    'SVM': SVC(
        class_weight='balanced', kernel='rbf', probability=True, random_state=SEED, C=1.0
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=SEED
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Out-of-fold predictions for train+val
oof_probs = np.zeros((len(y_trainval), len(base_models)))
# Test predictions (averaged across folds)
test_probs_l1 = np.zeros((len(y_test), len(base_models)))

for i, (name, model_template) in enumerate(base_models.items()):
    print(f"\n  {name}...")
    fold_test_probs = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(X_trainval, y_trainval)):
        X_ft, X_fv = X_trainval[fold_train_idx], X_trainval[fold_val_idx]
        y_ft, y_fv = y_trainval[fold_train_idx], y_trainval[fold_val_idx]

        # Clone model
        from sklearn.base import clone
        model = clone(model_template)
        model.fit(X_ft, y_ft)

        # OOF predictions
        oof_probs[fold_val_idx, i] = model.predict_proba(X_fv)[:, 1]

        # Test predictions
        fold_test_probs.append(model.predict_proba(X_test)[:, 1])

    test_probs_l1[:, i] = np.mean(fold_test_probs, axis=0)

    oof_f1 = f1_score(y_trainval, (oof_probs[:, i] > 0.5).astype(int), average='macro')
    oof_auc = roc_auc_score(y_trainval, oof_probs[:, i])
    print(f"    OOF F1={oof_f1:.3f} | OOF AUC={oof_auc:.3f}")

# ============================================================
# 3. ADD BERT PROBABILITY AS 5TH FEATURE
# ============================================================
print(f"\n{'='*60}")
print("ADDING BERT PROBABILITY")
print(f"{'='*60}")

# Try to load BERT results
bert_prob_available = False
try:
    with open(BERT_RESULTS_PATH, 'r') as f:
        bert_results = json.load(f)
    # BERT probability isn't saved per-sample, so we use BERT's test AUC as a signal
    # For proper stacking, we'd need per-sample BERT predictions
    # Workaround: use BERT fine-tuned model to generate probabilities
    # For now, we'll use the 4-model stack without BERT
    print("BERT results loaded but per-sample probabilities not available.")
    print("Using 4-model stacking (without BERT probability).")
    print("To add BERT: re-run script 04 to save per-sample predictions.")
except:
    print("BERT results not found. Using 4-model stacking.")

n_meta_features = len(base_models)
meta_train = oof_probs
meta_test = test_probs_l1

print(f"Meta-features: {n_meta_features} (one probability per base model)")
print(f"Meta train: {meta_train.shape} | Meta test: {meta_test.shape}")

# ============================================================
# 4. LEVEL 2: META-LEARNER WITH OPTUNA
# ============================================================
print(f"\n{'='*60}")
print("LEVEL 2: META-LEARNER TUNING")
print(f"{'='*60}")

if HAS_OPTUNA:
    def meta_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'scale_pos_weight': 1/scale_pw,
            'random_state': SEED, 'eval_metric': 'logloss', 'verbosity': 0,
        }
        cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, va_idx in cv_meta.split(meta_train, y_trainval):
            m = XGBClassifier(**params)
            m.fit(meta_train[tr_idx], y_trainval[tr_idx])
            p = m.predict(meta_train[va_idx])
            scores.append(f1_score(y_trainval[va_idx], p, average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(meta_objective, n_trials=80, show_progress_bar=True)
    best_meta_params = study.best_params
    print(f"Best meta CV F1: {study.best_value:.4f}")
else:
    best_meta_params = {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1,
                        'subsample': 0.8, 'colsample_bytree': 0.8}
    print("Using default meta-learner params (no Optuna)")

# ============================================================
# 5. TRAIN FINAL META-LEARNER AND EVALUATE
# ============================================================
print(f"\n{'='*60}")
print("FINAL STACKING ENSEMBLE (TEST SET)")
print(f"{'='*60}")

meta_model = XGBClassifier(**best_meta_params, scale_pos_weight=1/scale_pw,
                           random_state=SEED, eval_metric='logloss', verbosity=0)
meta_model.fit(meta_train, y_trainval)

stack_preds = meta_model.predict(meta_test)
stack_probs = meta_model.predict_proba(meta_test)[:, 1]

stack_acc = accuracy_score(y_test, stack_preds)
stack_f1 = f1_score(y_test, stack_preds, average='macro')
stack_auc = roc_auc_score(y_test, stack_probs)
stack_prec = precision_score(y_test, stack_preds, average='macro')
stack_rec = recall_score(y_test, stack_preds, average='macro')
stack_cm = confusion_matrix(y_test, stack_preds)
stack_spec = stack_cm[0,0] / (stack_cm[0,0] + stack_cm[0,1])
stack_sens = stack_cm[1,1] / (stack_cm[1,0] + stack_cm[1,1])

print(f"Accuracy:    {stack_acc:.3f}")
print(f"F1 (macro):  {stack_f1:.3f}")
print(f"ROC-AUC:     {stack_auc:.3f}")
print(f"Precision:   {stack_prec:.3f}")
print(f"Recall:      {stack_rec:.3f}")
print(f"Sensitivity: {stack_sens:.3f}")
print(f"Specificity: {stack_spec:.3f}")
print(f"Confusion Matrix:")
print(f"  TN={stack_cm[0,0]}  FP={stack_cm[0,1]}")
print(f"  FN={stack_cm[1,0]}  TP={stack_cm[1,1]}")

# Also compute simple averaging baseline
avg_probs = np.mean(test_probs_l1, axis=1)
avg_preds = (avg_probs > 0.5).astype(int)
avg_acc = accuracy_score(y_test, avg_preds)
avg_f1 = f1_score(y_test, avg_preds, average='macro')
avg_auc = roc_auc_score(y_test, avg_probs)

# Also get single best XGBoost (retrained on full trainval)
best_single = XGBClassifier(
    n_estimators=483, max_depth=7, learning_rate=0.288,
    min_child_weight=2, subsample=0.802, colsample_bytree=0.947,
    reg_alpha=0.136, reg_lambda=0.076, gamma=0.09,
    scale_pos_weight=1/scale_pw, random_state=SEED, eval_metric='logloss', verbosity=0
)
best_single.fit(X_trainval, y_trainval)
single_preds = best_single.predict(X_test)
single_probs = best_single.predict_proba(X_test)[:, 1]
single_acc = accuracy_score(y_test, single_preds)
single_f1 = f1_score(y_test, single_preds, average='macro')
single_auc = roc_auc_score(y_test, single_probs)

# ============================================================
# 6. COMPARISON TABLE
# ============================================================
print(f"\n{'='*60}")
print("FULL COMPARISON (TEST SET)")
print(f"{'='*60}")
print(f"{'Model':<45} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print(f"{'-'*65}")
print(f"{'XGBoost enhanced+tuned (single)':<45} {single_acc:>6.3f} {single_f1:>6.3f} {single_auc:>6.3f}")
print(f"{'Simple averaging (4 models)':<45} {avg_acc:>6.3f} {avg_f1:>6.3f} {avg_auc:>6.3f}")
print(f"{'STACKING ENSEMBLE (meta-XGBoost)':<45} {stack_acc:>6.3f} {stack_f1:>6.3f} {stack_auc:>6.3f}")

# Load previous results for full comparison
try:
    with open(f'{OUT_DIR}/multimodal_results.json', 'r') as f:
        mm = json.load(f)
    print(f"{'BERT fine-tuned':<45} {mm['bert_finetuned']['accuracy']:>6.3f} {mm['bert_finetuned']['f1_macro']:>6.3f} {mm['bert_finetuned']['roc_auc']:>6.3f}")
    print(f"{'BERT + acoustic fusion':<45} {mm['multimodal_fusion']['accuracy']:>6.3f} {mm['multimodal_fusion']['f1_macro']:>6.3f} {mm['multimodal_fusion']['roc_auc']:>6.3f}")
except:
    pass

# ============================================================
# 7. PLOTS
# ============================================================
colors = ['#C0392B', '#E67E22', '#2E75B6', '#27AE60']

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
model_names_plot = ['XGBoost\nsingle', 'Simple\naveraging', 'Stacking\nensemble']
metrics_vals = {
    'Accuracy': [single_acc, avg_acc, stack_acc],
    'F1 (macro)': [single_f1, avg_f1, stack_f1],
    'ROC-AUC': [single_auc, avg_auc, stack_auc],
}
x = np.arange(len(model_names_plot))
width = 0.25
for i, (metric, vals) in enumerate(metrics_vals.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric, color=colors[i], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Stacking Ensemble Comparison (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(model_names_plot, fontsize=11)
ax.set_ylim(0, 1.08)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/stacking_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs, color in [
    ('XGBoost single', single_probs, colors[0]),
    ('Simple averaging', avg_probs, colors[1]),
    ('Stacking ensemble', stack_probs, colors[2]),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Stacking Ensemble (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/stacking_roc.png', dpi=150, bbox_inches='tight')
plt.close()

# Meta-learner feature importance (which base model matters most)
meta_imp = pd.DataFrame({
    'model': list(base_models.keys()),
    'importance': meta_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(range(len(meta_imp)), meta_imp['importance'], color='#2E75B6', edgecolor='white')
ax.set_yticks(range(len(meta_imp)))
ax.set_yticklabels(meta_imp['model'], fontsize=11)
ax.invert_yaxis()
ax.set_title('Meta-learner: which base model matters most?', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/meta_learner_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(stack_cm, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Control', 'Dementia'], fontsize=10)
ax.set_yticklabels(['Control', 'Dementia'], fontsize=10)
ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('Actual', fontsize=11)
ax.set_title('Stacking Ensemble - Confusion Matrix', fontsize=12, fontweight='bold')
for r in range(2):
    for c in range(2):
        color = 'white' if stack_cm[r,c] > stack_cm.max()/2 else 'black'
        ax.text(c, r, str(stack_cm[r,c]), ha='center', va='center', fontsize=18, fontweight='bold', color=color)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/stacking_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 8. SAVE RESULTS
# ============================================================
results = {
    'single_xgboost': {'accuracy': single_acc, 'f1_macro': single_f1, 'roc_auc': single_auc},
    'simple_averaging': {'accuracy': avg_acc, 'f1_macro': avg_f1, 'roc_auc': avg_auc},
    'stacking_ensemble': {
        'accuracy': stack_acc, 'f1_macro': stack_f1, 'roc_auc': stack_auc,
        'precision_macro': stack_prec, 'recall_macro': stack_rec,
        'sensitivity': stack_sens, 'specificity': stack_spec,
        'confusion_matrix': stack_cm.tolist(),
        'meta_params': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                        for k, v in best_meta_params.items()},
    },
    'meta_feature_importance': dict(zip(meta_imp['model'], meta_imp['importance'].astype(float))),
}

with open(f'{OUT_DIR}/stacking_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {OUT_DIR}/:")
print(f"  stacking_comparison.png")
print(f"  stacking_roc.png")
print(f"  meta_learner_importance.png")
print(f"  stacking_confusion.png")
print(f"  stacking_results.json")
print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
