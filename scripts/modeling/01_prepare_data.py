"""
Phase 1: Data Preparation for Modeling
- Drop weak/redundant features (pause_rate, idea_density)
- One-hot encode task_folder
- Participant-level stratified train/val/test split (80/10/10)
- StandardScaler fit on train only
- Save all splits to CSV
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ============================================================
# 1. LOAD AND CLEAN FEATURES
# ============================================================
df = pd.read_csv('/mnt/user-data/uploads/features_all_clean.csv')
print(f"Loaded: {df.shape[0]} records, {df.shape[1]} columns")

# Drop weak/redundant features identified in EDA
drop_features = ['pause_rate', 'idea_density']
df = df.drop(columns=drop_features)
print(f"Dropped: {drop_features}")

# Define feature columns (18 - 2 dropped = 16 numeric features)
meta_cols = ['participant_id', 'session', 'task_folder', 'diagnosis', 'group', 'label', 'word_count']
numeric_features = [c for c in df.columns if c not in meta_cols]
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")

# One-hot encode task_folder
task_dummies = pd.get_dummies(df['task_folder'], prefix='task', dtype=int)
print(f"Task dummies: {list(task_dummies.columns)}")

# Combine
df = pd.concat([df, task_dummies], axis=1)
feature_cols = numeric_features + list(task_dummies.columns)
print(f"\nTotal features for modeling: {len(feature_cols)}")
print(f"  Numeric: {len(numeric_features)}")
print(f"  Task dummies: {len(task_dummies.columns)}")

# ============================================================
# 2. PARTICIPANT-LEVEL STRATIFIED SPLIT
# ============================================================
# Get participant-level info
participant_df = df.groupby('participant_id').agg(
    group=('group', 'first'),
    label=('label', 'first'),
    n_samples=('label', 'count')
).reset_index()

print(f"\n{'='*60}")
print(f"PARTICIPANT-LEVEL SPLIT")
print(f"{'='*60}")
print(f"Total participants: {len(participant_df)}")
print(f"  Control:  {(participant_df['group']=='Control').sum()}")
print(f"  Dementia: {(participant_df['group']=='Dementia').sum()}")

# First split: 80% train, 20% temp (will become 10% val + 10% test)
train_pids, temp_pids = train_test_split(
    participant_df['participant_id'],
    test_size=0.20,
    random_state=42,
    stratify=participant_df['group']
)

# Second split: 50/50 of temp → 10% val, 10% test
temp_df = participant_df[participant_df['participant_id'].isin(temp_pids)]
val_pids, test_pids = train_test_split(
    temp_df['participant_id'],
    test_size=0.50,
    random_state=42,
    stratify=temp_df['group']
)

# Assign split labels
df['split'] = 'train'
df.loc[df['participant_id'].isin(val_pids), 'split'] = 'val'
df.loc[df['participant_id'].isin(test_pids), 'split'] = 'test'

# ============================================================
# 3. VERIFY SPLIT INTEGRITY
# ============================================================
print(f"\n{'='*60}")
print(f"SPLIT VERIFICATION")
print(f"{'='*60}")

for split_name in ['train', 'val', 'test']:
    split_df = df[df['split'] == split_name]
    n_participants = split_df['participant_id'].nunique()
    n_samples = len(split_df)
    n_control = (split_df['group'] == 'Control').sum()
    n_dementia = (split_df['group'] == 'Dementia').sum()
    pct_dementia = n_dementia / n_samples * 100
    print(f"\n{split_name.upper():>5}: {n_participants:>3} participants, {n_samples:>4} samples")
    print(f"       Control={n_control}, Dementia={n_dementia} ({pct_dementia:.1f}% Dementia)")

# Verify NO participant leakage
train_set = set(df[df['split']=='train']['participant_id'])
val_set = set(df[df['split']=='val']['participant_id'])
test_set = set(df[df['split']=='test']['participant_id'])

assert len(train_set & val_set) == 0, "LEAKAGE: train-val overlap!"
assert len(train_set & test_set) == 0, "LEAKAGE: train-test overlap!"
assert len(val_set & test_set) == 0, "LEAKAGE: val-test overlap!"
print(f"\n✅ NO participant leakage detected!")
print(f"   Train ∩ Val  = {len(train_set & val_set)} participants")
print(f"   Train ∩ Test = {len(train_set & test_set)} participants")
print(f"   Val ∩ Test   = {len(val_set & test_set)} participants")

# ============================================================
# 4. PREPARE X, y AND SCALE
# ============================================================
X_train = df[df['split']=='train'][feature_cols].values
X_val   = df[df['split']=='val'][feature_cols].values
X_test  = df[df['split']=='test'][feature_cols].values

y_train = df[df['split']=='train']['label'].values
y_val   = df[df['split']=='val']['label'].values
y_test  = df[df['split']=='test']['label'].values

# Scale numeric features only (not task dummies)
scaler = StandardScaler()
n_numeric = len(numeric_features)

X_train[:, :n_numeric] = scaler.fit_transform(X_train[:, :n_numeric])
X_val[:, :n_numeric]   = scaler.transform(X_val[:, :n_numeric])
X_test[:, :n_numeric]  = scaler.transform(X_test[:, :n_numeric])

print(f"\n{'='*60}")
print(f"FINAL ARRAYS")
print(f"{'='*60}")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape} (pos rate: {y_train.mean():.3f})")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}   (pos rate: {y_val.mean():.3f})")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}  (pos rate: {y_test.mean():.3f})")

# ============================================================
# 5. SAVE EVERYTHING
# ============================================================
out_dir = '/home/claude/modeling'
os.makedirs(out_dir, exist_ok=True)

# Save split assignments
df.to_csv(f'{out_dir}/features_with_splits.csv', index=False)

# Save numpy arrays
np.savez(f'{out_dir}/split_data.npz',
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_train=y_train, y_val=y_val, y_test=y_test,
    feature_names=np.array(feature_cols),
    numeric_feature_names=np.array(numeric_features)
)

# Save scaler params for reproducibility
scaler_df = pd.DataFrame({
    'feature': numeric_features,
    'mean': scaler.mean_,
    'std': scaler.scale_
})
scaler_df.to_csv(f'{out_dir}/scaler_params.csv', index=False)

print(f"\nSaved to {out_dir}/:")
print(f"  features_with_splits.csv")
print(f"  split_data.npz")
print(f"  scaler_params.csv")
print(f"\nFeature columns ({len(feature_cols)}):")
for i, f in enumerate(feature_cols):
    print(f"  [{i:2d}] {f}")
