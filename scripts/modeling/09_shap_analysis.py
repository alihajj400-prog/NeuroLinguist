"""
Script 09: SHAP Analysis for Model Interpretability
=====================================================
1. SHAP values for best XGBoost model (enhanced features)
2. Global feature importance (beeswarm plot)
3. Per-class SHAP analysis (what drives Control vs Dementia predictions)
4. Sample-level explanations (waterfall plots for interesting cases)
5. SHAP interaction effects

Paths: D:/FYP/ (ready to run, no edits needed)
Requirements: pip install shap
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import shap

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'D:/FYP/data/features/features_all_clean.csv'
TRANSCRIPTS_PATH = 'D:/FYP/data/transcripts/transcripts_clean.csv'
OUT_DIR = 'D:/FYP/results/SHAP Analysis'
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 1. LOAD AND PREPARE (same pipeline)
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

df['filler_rate'] = df['filler_count'] / df['word_count'].clip(lower=1)
df['unfilled_pause_rate'] = df['unfilled_pause_count'] / df['word_count'].clip(lower=1)
df['repetition_rate_bio'] = df['repetition_count'] / df['word_count'].clip(lower=1)
df['revision_rate'] = df['revision_count'] / df['word_count'].clip(lower=1)
df['fragment_rate'] = df['fragment_count'] / df['word_count'].clip(lower=1)
df['utterance_rate'] = df['utterance_count'] / df['word_count'].clip(lower=1)
df['words_per_utterance'] = df['word_count'] / df['utterance_count'].clip(lower=1)

derived_cols = ['filler_rate', 'unfilled_pause_rate', 'repetition_rate_bio',
                'revision_rate', 'fragment_rate', 'utterance_rate', 'words_per_utterance']

task_dependent = ['information_unit_coverage', 'story_recall_similarity',
                  'syntactic_complexity', 'semantic_coherence']
for feat in task_dependent:
    if feat in df.columns:
        df[f'{feat}_task_z'] = df.groupby('task_folder')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        )
task_z_cols = [f'{f}_task_z' for f in task_dependent if f in df.columns]

task_dummies = pd.get_dummies(df['task_folder'], prefix='task', dtype=int)
df = pd.concat([df, task_dummies], axis=1)
task_cols = list(task_dummies.columns)

original_numeric = ['syntactic_complexity', 'pronoun_noun_ratio', 'repetition_rate',
    'word_frequency_index', 'content_function_ratio', 'mean_pause_duration',
    'speech_time_ratio', 'pitch_range', 'pitch_variability', 'intensity_range',
    'jitter', 'shimmer', 'semantic_coherence', 'information_unit_coverage',
    'story_recall_similarity', 'global_semantic_drift']
enhanced_numeric = original_numeric + derived_cols + task_z_cols + ['word_count']
enhanced_features = enhanced_numeric + task_cols

# Split (same seed)
participant_df = df.groupby('participant_id').agg(group=('group', 'first'), label=('label', 'first')).reset_index()
train_pids, temp_pids = train_test_split(participant_df['participant_id'], test_size=0.20, random_state=SEED, stratify=participant_df['group'])
temp_df_p = participant_df[participant_df['participant_id'].isin(temp_pids)]
val_pids, test_pids = train_test_split(temp_df_p['participant_id'], test_size=0.50, random_state=SEED, stratify=temp_df_p['group'])

df['split'] = 'train'
df.loc[df['participant_id'].isin(val_pids), 'split'] = 'val'
df.loc[df['participant_id'].isin(test_pids), 'split'] = 'test'

trainval_df = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

# Prepare arrays
X_tv = trainval_df[enhanced_features].values.astype(float)
X_test = test_df[enhanced_features].values.astype(float)
y_tv = trainval_df['label'].values
y_test = test_df['label'].values

np.nan_to_num(X_tv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

n_num = len(enhanced_numeric)
scaler = StandardScaler()
X_tv[:, :n_num] = scaler.fit_transform(X_tv[:, :n_num])
X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

# Make DataFrames with feature names for SHAP
X_tv_df = pd.DataFrame(X_tv, columns=enhanced_features)
X_test_df = pd.DataFrame(X_test, columns=enhanced_features)

print(f"Train+Val: {X_tv.shape} | Test: {X_test.shape}")

# ============================================================
# 2. TRAIN BEST XGBOOST MODEL
# ============================================================
print(f"\n{'='*60}")
print("TRAINING XGBOOST")
print(f"{'='*60}")

spw = (y_tv == 0).sum() / max((y_tv == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=483, max_depth=7, learning_rate=0.288,
    min_child_weight=2, subsample=0.802, colsample_bytree=0.947,
    reg_alpha=0.136, reg_lambda=0.076, gamma=0.09,
    scale_pos_weight=1/spw, random_state=SEED, eval_metric='logloss', verbosity=0
)
model.fit(X_tv, y_tv)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')
print(f"Test Acc={acc:.3f} | F1={f1:.3f}")

# ============================================================
# 3. COMPUTE SHAP VALUES
# ============================================================
print(f"\n{'='*60}")
print("COMPUTING SHAP VALUES")
print(f"{'='*60}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Expected value (base rate): {explainer.expected_value:.4f}")

# ============================================================
# 4. GLOBAL PLOTS
# ============================================================
print(f"\n{'='*60}")
print("GENERATING SHAP PLOTS")
print(f"{'='*60}")

# 4a. Beeswarm / Summary plot
print("  Creating beeswarm plot...")
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_test_df, show=False, max_display=20)
plt.title('SHAP Feature Importance (Beeswarm Plot)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("    Saved: shap_beeswarm.png")

# 4b. Bar plot (mean absolute SHAP)
print("  Creating bar plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, plot_type='bar', show=False, max_display=20)
plt.title('Mean |SHAP Value| (Global Feature Importance)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("    Saved: shap_bar.png")

# ============================================================
# 5. PER-CLASS ANALYSIS
# ============================================================
print("  Analyzing per-class SHAP...")

# Separate by true class
control_mask = y_test == 0
dementia_mask = y_test == 1

mean_shap_control = np.mean(np.abs(shap_values[control_mask]), axis=0)
mean_shap_dementia = np.mean(np.abs(shap_values[dementia_mask]), axis=0)

shap_comparison = pd.DataFrame({
    'feature': enhanced_features,
    'mean_abs_shap_control': mean_shap_control,
    'mean_abs_shap_dementia': mean_shap_dementia,
    'mean_abs_shap_overall': np.mean(np.abs(shap_values), axis=0),
}).sort_values('mean_abs_shap_overall', ascending=False)

print(f"\n  Top 10 features by SHAP importance:")
for _, row in shap_comparison.head(10).iterrows():
    print(f"    {row['feature']:<35} Overall={row['mean_abs_shap_overall']:.4f} | "
          f"Control={row['mean_abs_shap_control']:.4f} | Dementia={row['mean_abs_shap_dementia']:.4f}")

shap_comparison.to_csv(f'{OUT_DIR}/shap_feature_importance.csv', index=False)

# Per-class bar plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top15 = shap_comparison.head(15)

ax = axes[0]
ax.barh(range(len(top15)), top15['mean_abs_shap_control'], color='#27AE60', edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Control Group\nMean |SHAP|', fontsize=12, fontweight='bold')
ax.set_xlabel('Mean |SHAP value|')

ax = axes[1]
ax.barh(range(len(top15)), top15['mean_abs_shap_dementia'], color='#C0392B', edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Dementia Group\nMean |SHAP|', fontsize=12, fontweight='bold')
ax.set_xlabel('Mean |SHAP value|')

plt.suptitle('SHAP Importance by Diagnostic Group', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_per_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: shap_per_class.png")

# ============================================================
# 6. WATERFALL PLOTS FOR INTERESTING CASES
# ============================================================
print("  Creating waterfall plots for sample cases...")

# Find interesting cases
probs = model.predict_proba(X_test)[:, 1]

# True positive with high confidence
tp_mask = (preds == 1) & (y_test == 1)
tp_idx = np.where(tp_mask)[0]
if len(tp_idx) > 0:
    tp_most_confident = tp_idx[np.argmax(probs[tp_idx])]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[tp_most_confident],
        base_values=explainer.expected_value,
        data=X_test_df.iloc[tp_most_confident],
        feature_names=enhanced_features
    ), show=False, max_display=12)
    plt.title(f'True Positive (Dementia, confidence={probs[tp_most_confident]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/shap_waterfall_true_positive.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    Saved: shap_waterfall_true_positive.png")

# True negative with high confidence
tn_mask = (preds == 0) & (y_test == 0)
tn_idx = np.where(tn_mask)[0]
if len(tn_idx) > 0:
    tn_most_confident = tn_idx[np.argmin(probs[tn_idx])]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[tn_most_confident],
        base_values=explainer.expected_value,
        data=X_test_df.iloc[tn_most_confident],
        feature_names=enhanced_features
    ), show=False, max_display=12)
    plt.title(f'True Negative (Control, confidence={1-probs[tn_most_confident]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/shap_waterfall_true_negative.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    Saved: shap_waterfall_true_negative.png")

# False positive (Control misclassified as Dementia)
fp_mask = (preds == 1) & (y_test == 0)
fp_idx = np.where(fp_mask)[0]
if len(fp_idx) > 0:
    fp_sample = fp_idx[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[fp_sample],
        base_values=explainer.expected_value,
        data=X_test_df.iloc[fp_sample],
        feature_names=enhanced_features
    ), show=False, max_display=12)
    plt.title(f'False Positive (Control called Dementia, prob={probs[fp_sample]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/shap_waterfall_false_positive.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    Saved: shap_waterfall_false_positive.png")

# False negative (Dementia misclassified as Control)
fn_mask = (preds == 0) & (y_test == 1)
fn_idx = np.where(fn_mask)[0]
if len(fn_idx) > 0:
    fn_sample = fn_idx[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[fn_sample],
        base_values=explainer.expected_value,
        data=X_test_df.iloc[fn_sample],
        feature_names=enhanced_features
    ), show=False, max_display=12)
    plt.title(f'False Negative (Dementia called Control, prob={probs[fn_sample]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/shap_waterfall_false_negative.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    Saved: shap_waterfall_false_negative.png")

# ============================================================
# 7. SHAP DEPENDENCE PLOTS (top 4 features)
# ============================================================
print("  Creating dependence plots...")

top4 = shap_comparison.head(4)['feature'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (feat, ax) in enumerate(zip(top4, axes.flat)):
    feat_idx = enhanced_features.index(feat)
    shap.dependence_plot(feat_idx, shap_values, X_test_df, ax=ax, show=False)
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')

plt.suptitle('SHAP Dependence Plots (Top 4 Features)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("    Saved: shap_dependence.png")

# ============================================================
# 8. FEATURE DIRECTION ANALYSIS
# ============================================================
print(f"\n{'='*60}")
print("FEATURE DIRECTION ANALYSIS")
print(f"{'='*60}")

# For each feature, determine: does higher value push toward Dementia or Control?
directions = []
for i, feat in enumerate(enhanced_features):
    feat_vals = X_test[:, i]
    feat_shap = shap_values[:, i]
    correlation = np.corrcoef(feat_vals, feat_shap)[0, 1]
    direction = "Higher → Dementia" if correlation > 0 else "Higher → Control"
    directions.append({
        'feature': feat,
        'correlation_with_shap': correlation,
        'direction': direction,
        'mean_abs_shap': np.mean(np.abs(feat_shap)),
    })

dir_df = pd.DataFrame(directions).sort_values('mean_abs_shap', ascending=False)
dir_df.to_csv(f'{OUT_DIR}/shap_directions.csv', index=False)

print(f"\nTop 15 features with clinical direction:")
for _, row in dir_df.head(15).iterrows():
    arrow = "↑ Dem" if row['correlation_with_shap'] > 0 else "↓ Dem"
    print(f"  {row['feature']:<35} SHAP={row['mean_abs_shap']:.4f} | {arrow} (r={row['correlation_with_shap']:.3f})")

# ============================================================
# 9. SAVE SUMMARY
# ============================================================
summary = {
    'model': 'XGBoost (enhanced, tuned)',
    'test_accuracy': acc,
    'test_f1': f1,
    'expected_value': float(explainer.expected_value),
    'top_10_features': shap_comparison.head(10)[['feature', 'mean_abs_shap_overall']].to_dict('records'),
    'n_true_positive': int(tp_mask.sum()),
    'n_true_negative': int(tn_mask.sum()),
    'n_false_positive': int(fp_mask.sum()),
    'n_false_negative': int(fn_mask.sum()),
}

with open(f'{OUT_DIR}/shap_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"ALL SHAP OUTPUTS SAVED TO {OUT_DIR}/")
print(f"{'='*60}")
print(f"  shap_beeswarm.png         - Global feature importance with direction")
print(f"  shap_bar.png              - Mean |SHAP| bar chart")
print(f"  shap_per_class.png        - Importance by diagnostic group")
print(f"  shap_waterfall_*.png      - Per-sample explanations (TP, TN, FP, FN)")
print(f"  shap_dependence.png       - Feature value vs SHAP value")
print(f"  shap_feature_importance.csv")
print(f"  shap_directions.csv")
print(f"  shap_summary.json")
print(f"\nDONE")
