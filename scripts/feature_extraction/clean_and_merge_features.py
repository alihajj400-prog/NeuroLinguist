"""
Clean and Merge All Features
FYP: AI-Based Alzheimer's Detection

Actions:
1. Inner join linguistic, acoustic, semantic on (participant_id, session, task_folder)
2. Cap outliers at 99th percentile
3. Remove records with critical zeros
4. Drop redundant features (mean_sentence_length, lexical_diversity_mattr, dependency_distance)
5. Output features_all_clean.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def clean_and_merge(ling_path: str, acou_path: str, sem_path: str, output_dir: str):
    print("=" * 70)
    print("CLEAN AND MERGE ALL FEATURES")
    print("=" * 70)
    
    # Load all three
    ling = pd.read_csv(ling_path)
    acou = pd.read_csv(acou_path)
    sem = pd.read_csv(sem_path)
    
    print(f"\n📂 Loaded:")
    print(f"   Linguistic: {len(ling)} records")
    print(f"   Acoustic:   {len(acou)} records")
    print(f"   Semantic:   {len(sem)} records")
    
    # =========================================================================
    # 1. MERGE ON KEYS
    # =========================================================================
    merge_keys = ['participant_id', 'session', 'task_folder']
    
    # First merge linguistic + acoustic
    df = pd.merge(ling, acou, on=merge_keys, suffixes=('', '_acou'))
    
    # Then merge with semantic
    df = pd.merge(df, sem, on=merge_keys, suffixes=('', '_sem'))
    
    print(f"\n🔗 After inner join: {len(df)} records")
    
    # Keep only one diagnosis column
    if 'diagnosis_acou' in df.columns:
        df = df.drop(columns=['diagnosis_acou'])
    if 'diagnosis_sem' in df.columns:
        df = df.drop(columns=['diagnosis_sem'])
    
    # =========================================================================
    # 2. DROP REDUNDANT LINGUISTIC FEATURES
    # =========================================================================
    drop_ling = ['mean_sentence_length', 'lexical_diversity_mattr', 'dependency_distance']
    dropped = []
    for col in drop_ling:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
    
    print(f"\n🗑️  Dropped redundant features: {dropped}")
    
    # =========================================================================
    # 3. REMOVE RECORDS WITH CRITICAL ZEROS
    # =========================================================================
    # Remove if word_count < 10
    if 'word_count' in df.columns:
        before = len(df)
        df = df[df['word_count'] >= 10]
        print(f"\n🗑️  Removed {before - len(df)} records with word_count < 10")
    
    # Remove if pause_rate = 0 (indicates failed extraction)
    before = len(df)
    df = df[df['pause_rate'] > 0]
    print(f"🗑️  Removed {before - len(df)} records with pause_rate = 0")
    
    # =========================================================================
    # 4. CAP OUTLIERS AT 99TH PERCENTILE
    # =========================================================================
    # Features to cap
    cap_features = [
        'pronoun_noun_ratio', 'content_function_ratio',  # Linguistic
        'mean_pause_duration', 'intensity_range', 'pitch_range',  # Acoustic
    ]
    
    print(f"\n📊 Capping outliers at 99th percentile:")
    for feat in cap_features:
        if feat in df.columns:
            q99 = df[feat].quantile(0.99)
            n_capped = (df[feat] > q99).sum()
            df[feat] = df[feat].clip(upper=q99)
            if n_capped > 0:
                print(f"   {feat}: capped {n_capped} values at {q99:.4f}")
    
    # =========================================================================
    # 5. CREATE BINARY LABEL
    # =========================================================================
    df['group'] = df['diagnosis'].apply(lambda x: 'Control' if x == 'Control' else 'Dementia')
    df['label'] = (df['group'] == 'Dementia').astype(int)
    
    # =========================================================================
    # 6. DEFINE FINAL FEATURE COLUMNS
    # =========================================================================
    id_cols = ['participant_id', 'session', 'task_folder', 'diagnosis', 'group', 'label']
    
    ling_features = ['syntactic_complexity', 'pronoun_noun_ratio', 'repetition_rate',
                     'word_frequency_index', 'content_function_ratio']
    
    acou_features = ['pause_rate', 'mean_pause_duration', 'speech_time_ratio',
                     'pitch_range', 'pitch_variability', 'intensity_range', 'jitter', 'shimmer']
    
    sem_features = ['semantic_coherence', 'information_unit_coverage', 
                    'story_recall_similarity', 'idea_density', 'global_semantic_drift']
    
    all_features = ling_features + acou_features + sem_features
    
    # Check which features exist
    existing_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
    
    # Keep only relevant columns
    keep_cols = id_cols + existing_features + ['word_count']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final records: {len(df)}")
    print(f"Final features: {len(existing_features)}")
    print(f"\nFeature breakdown:")
    print(f"   Linguistic (5): {[f for f in ling_features if f in existing_features]}")
    print(f"   Acoustic (8):   {[f for f in acou_features if f in existing_features]}")
    print(f"   Semantic (5):   {[f for f in sem_features if f in existing_features]}")
    
    print(f"\n📈 Class distribution:")
    print(df['group'].value_counts())
    
    print(f"\n📊 Feature statistics:")
    print(df[existing_features].describe().round(4).T[['mean', 'std', 'min', 'max']])
    
    # =========================================================================
    # 8. SAVE
    # =========================================================================
    output_path = Path(output_dir) / 'features_all_clean.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and merge all features")
    parser.add_argument("--linguistic", type=str, default="D:/FYP/data/features/features_linguistic.csv")
    parser.add_argument("--acoustic", type=str, default="D:/FYP/data/features/features_acoustic.csv")
    parser.add_argument("--semantic", type=str, default="D:/FYP/data/features/features_semantic.csv")
    parser.add_argument("--output", type=str, default="D:/FYP/data/features")
    
    args = parser.parse_args()
    clean_and_merge(args.linguistic, args.acoustic, args.semantic, args.output)
