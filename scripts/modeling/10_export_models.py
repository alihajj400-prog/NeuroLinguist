"""
Script 10: Export Trained Models for Deployment
=================================================
Trains and saves all models needed for the Dockerized app:
- Cookie specialist (4 models)
- Non-cookie specialist (4 models)
- Scalers
- Feature lists

Paths: D:/FYP/ (ready to run)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import clone

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'D:/FYP/data/features/features_all_clean.csv'
TRANSCRIPTS_PATH = 'D:/FYP/data/transcripts/transcripts_clean.csv'
MODEL_DIR = 'D:/FYP/models/deployment'
SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 1. LOAD AND PREPARE (same pipeline as script 07)
# ============================================================
print("Loading data...")
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

task_dependent = ['information_unit_coverage', 'story_recall_similarity',
                  'syntactic_complexity', 'semantic_coherence']
# Store task-level means and stds for deployment
task_stats = {}
for feat in task_dependent:
    if feat in df.columns:
        stats = df.groupby('task_folder')[feat].agg(['mean', 'std']).to_dict('index')
        task_stats[feat] = stats
        df[f'{feat}_task_z'] = df.groupby('task_folder')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        )

task_z_cols = [f'{f}_task_z' for f in task_dependent if f in df.columns]
derived_cols = ['filler_rate', 'unfilled_pause_rate', 'repetition_rate_bio',
                'revision_rate', 'fragment_rate', 'utterance_rate', 'words_per_utterance']

original_numeric = ['syntactic_complexity', 'pronoun_noun_ratio', 'repetition_rate',
    'word_frequency_index', 'content_function_ratio', 'mean_pause_duration',
    'speech_time_ratio', 'pitch_range', 'pitch_variability', 'intensity_range',
    'jitter', 'shimmer', 'semantic_coherence', 'information_unit_coverage',
    'story_recall_similarity', 'global_semantic_drift']

enhanced_numeric = original_numeric + derived_cols + task_z_cols + ['word_count']

# Cookie features (no task dummies needed)
cookie_features_numeric = enhanced_numeric
cookie_features = cookie_features_numeric

# Non-cookie features
noncookie_numeric = [f for f in enhanced_numeric if f not in ['information_unit_coverage', 'story_recall_similarity',
                     'information_unit_coverage_task_z', 'story_recall_similarity_task_z']]
task_cols = ['task_cookie', 'task_fluency', 'task_recall', 'task_sentence']
noncookie_features = noncookie_numeric + task_cols

# Add task dummies
for tc in task_cols:
    task_name = tc.replace('task_', '')
    df[tc] = (df['task_folder'] == task_name).astype(int)

# ============================================================
# 2. SPLIT (use ALL data for training — train+val+test)
# ============================================================
# For deployment, we train on ALL available data
print("Training on full dataset for deployment...")

cookie_df = df[df['task_folder'] == 'cookie'].reset_index(drop=True)
noncookie_df = df[df['task_folder'] != 'cookie'].reset_index(drop=True)

print(f"  Cookie: {len(cookie_df)} samples")
print(f"  Non-cookie: {len(noncookie_df)} samples")

# ============================================================
# 3. TRAIN AND SAVE COOKIE SPECIALIST
# ============================================================
print("\nTraining cookie specialist...")

X_cookie = cookie_df[cookie_features].values.astype(float)
y_cookie = cookie_df['label'].values
np.nan_to_num(X_cookie, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

cookie_scaler = StandardScaler()
n_c = len(cookie_features_numeric)
X_cookie[:, :n_c] = cookie_scaler.fit_transform(X_cookie[:, :n_c])

spw_c = (y_cookie == 0).sum() / max((y_cookie == 1).sum(), 1)

cookie_models = {
    'xgboost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=1/spw_c, random_state=SEED, eval_metric='logloss', verbosity=0),
    'rf': RandomForestClassifier(n_estimators=500, class_weight='balanced',
        min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    'svm': SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=SEED),
    'lr': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
}

for name, model in cookie_models.items():
    model.fit(X_cookie, y_cookie)
    joblib.dump(model, f'{MODEL_DIR}/cookie_{name}.joblib')
    print(f"  Saved cookie_{name}.joblib")

joblib.dump(cookie_scaler, f'{MODEL_DIR}/cookie_scaler.joblib')
print("  Saved cookie_scaler.joblib")

# ============================================================
# 4. TRAIN AND SAVE NON-COOKIE SPECIALIST
# ============================================================
print("\nTraining non-cookie specialist...")

X_noncookie = noncookie_df[noncookie_features].values.astype(float)
y_noncookie = noncookie_df['label'].values
np.nan_to_num(X_noncookie, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

noncookie_scaler = StandardScaler()
n_nc = len(noncookie_numeric)
X_noncookie[:, :n_nc] = noncookie_scaler.fit_transform(X_noncookie[:, :n_nc])

spw_nc = (y_noncookie == 0).sum() / max((y_noncookie == 1).sum(), 1)

noncookie_models = {
    'xgboost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=1/spw_nc, random_state=SEED, eval_metric='logloss', verbosity=0),
    'rf': RandomForestClassifier(n_estimators=500, class_weight='balanced',
        min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    'svm': SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=SEED),
    'lr': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
}

for name, model in noncookie_models.items():
    model.fit(X_noncookie, y_noncookie)
    joblib.dump(model, f'{MODEL_DIR}/noncookie_{name}.joblib')
    print(f"  Saved noncookie_{name}.joblib")

joblib.dump(noncookie_scaler, f'{MODEL_DIR}/noncookie_scaler.joblib')
print("  Saved noncookie_scaler.joblib")

# ============================================================
# 5. SAVE FEATURE CONFIGS
# ============================================================
config = {
    'cookie_features': cookie_features,
    'cookie_numeric': cookie_features_numeric,
    'noncookie_features': noncookie_features,
    'noncookie_numeric': noncookie_numeric,
    'task_cols': task_cols,
    'task_stats': {feat: {task: {'mean': float(s['mean']), 'std': float(s['std'])}
                          for task, s in stats.items()}
                   for feat, stats in task_stats.items()},
    'model_names': ['xgboost', 'rf', 'svm', 'lr'],
}

with open(f'{MODEL_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("\nSaved config.json")

# ============================================================
# 6. VERIFY
# ============================================================
print(f"\n{'='*50}")
print(f"ALL MODELS SAVED TO {MODEL_DIR}/")
print(f"{'='*50}")

for f_name in os.listdir(MODEL_DIR):
    size = os.path.getsize(f'{MODEL_DIR}/{f_name}')
    print(f"  {f_name:<35} {size/1024:.0f} KB")

print(f"\nTotal files: {len(os.listdir(MODEL_DIR))}")
print("DONE")
