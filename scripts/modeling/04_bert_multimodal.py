"""
Script 04: BERT Fine-Tuning + Multi-Modal Fusion
=================================================
Stage A: Fine-tune BERT on transcript text (text-only classifier)
Stage B: Extract [CLS] embeddings from fine-tuned BERT
Stage C: Fuse [CLS] + acoustic features → XGBoost (multi-modal)
Stage D: Compare all approaches

Paths: D:/FYP/ (ready to run, no edits needed)
Requirements: pip install transformers torch scikit-learn xgboost pandas numpy matplotlib
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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================
FEATURES_PATH = 'D:/FYP/data/features/features_all_clean.csv'
TRANSCRIPTS_PATH = 'D:/FYP/data/transcripts/transcripts_clean.csv'
SPLITS_PATH = 'D:/FYP/data/features/features_with_splits.csv'
OUT_DIR = 'D:/FYP/results/Modeling Results'
MODEL_DIR = 'D:/FYP/models'

BERT_MODEL = 'bert-base-uncased'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. LOAD AND MERGE DATA
# ============================================================
print(f"\n{'='*60}")
print("LOADING DATA")
print(f"{'='*60}")

# Load features (has the splits already assigned from script 01)
splits_df = pd.read_csv(SPLITS_PATH, encoding='latin-1')
print(f"Features with splits: {splits_df.shape}")

# Load transcripts
transcripts_df = pd.read_csv(TRANSCRIPTS_PATH, encoding='latin-1')
print(f"Transcripts: {transcripts_df.shape}")

# Create join keys
splits_df['key'] = splits_df['participant_id'].astype(str) + '_' + splits_df['session'].astype(str) + '_' + splits_df['task_folder']
transcripts_df['key'] = transcripts_df['participant_id'].astype(str) + '_' + transcripts_df['session'].astype(str) + '_' + transcripts_df['task_folder']

# Merge: get text + split + label + acoustic features
acoustic_cols = ['mean_pause_duration', 'speech_time_ratio', 'pitch_range',
                 'pitch_variability', 'intensity_range', 'jitter', 'shimmer']

merge_cols_splits = ['key', 'split', 'label'] + [c for c in acoustic_cols if c in splits_df.columns]
merge_cols_text = ['key', 'cleaned_text']

df = pd.merge(
    splits_df[merge_cols_splits],
    transcripts_df[merge_cols_text],
    on='key', how='inner'
)

# Drop rows with missing text
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.len() > 10]

print(f"Merged dataset: {len(df)} records")
print(f"  Train: {(df['split']=='train').sum()}")
print(f"  Val:   {(df['split']=='val').sum()}")
print(f"  Test:  {(df['split']=='test').sum()}")

# Split into sets
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} (Dementia={train_df['label'].sum()})")
print(f"Val:   {len(val_df)} (Dementia={val_df['label'].sum()})")
print(f"Test:  {len(test_df)} (Dementia={test_df['label'].sum()})")

# ============================================================
# 2. BERT DATASET AND TOKENIZER
# ============================================================
print(f"\n{'='*60}")
print("TOKENIZING")
print(f"{'='*60}")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TranscriptDataset(train_df['cleaned_text'].values, train_df['label'].values, tokenizer, MAX_LEN)
val_dataset = TranscriptDataset(val_df['cleaned_text'].values, val_df['label'].values, tokenizer, MAX_LEN)
test_dataset = TranscriptDataset(test_df['cleaned_text'].values, test_df['label'].values, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Tokenizer: {BERT_MODEL}")
print(f"Max length: {MAX_LEN}")
print(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

# ============================================================
# 3. STAGE A: FINE-TUNE BERT
# ============================================================
print(f"\n{'='*60}")
print("STAGE A: FINE-TUNING BERT")
print(f"{'='*60}")

# Handle class imbalance with weighted loss
n_pos = train_df['label'].sum()
n_neg = len(train_df) - n_pos
weight = torch.tensor([n_pos / len(train_df), n_neg / len(train_df)], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weight)

model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

best_val_f1 = 0
train_losses = []
val_f1s = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    val_preds, val_true, val_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_f1 = f1_score(val_true, val_preds, average='macro')
    val_acc = accuracy_score(val_true, val_preds)
    val_auc = roc_auc_score(val_true, val_probs)
    val_f1s.append(val_f1)

    print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f} | Val AUC: {val_auc:.3f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), f'{MODEL_DIR}/bert_best.pt')
        print(f"    → Saved best model (F1={val_f1:.3f})")

# Load best model
model.load_state_dict(torch.load(f'{MODEL_DIR}/bert_best.pt', weights_only=True))
print(f"\nBest validation F1: {best_val_f1:.3f}")

# ============================================================
# 4. EVALUATE BERT-ONLY ON TEST SET
# ============================================================
print(f"\n{'='*60}")
print("BERT-ONLY TEST EVALUATION")
print(f"{'='*60}")

model.eval()
test_preds, test_true, test_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

bert_acc = accuracy_score(test_true, test_preds)
bert_f1 = f1_score(test_true, test_preds, average='macro')
bert_auc = roc_auc_score(test_true, test_probs)
bert_prec = precision_score(test_true, test_preds, average='macro')
bert_rec = recall_score(test_true, test_preds, average='macro')
bert_cm = confusion_matrix(test_true, test_preds)

print(f"BERT-only Test Results:")
print(f"  Accuracy:  {bert_acc:.3f}")
print(f"  F1 (macro):{bert_f1:.3f}")
print(f"  ROC-AUC:   {bert_auc:.3f}")
print(f"  Precision: {bert_prec:.3f}")
print(f"  Recall:    {bert_rec:.3f}")
print(f"  Confusion Matrix:")
print(f"    TN={bert_cm[0,0]}  FP={bert_cm[0,1]}")
print(f"    FN={bert_cm[1,0]}  TP={bert_cm[1,1]}")

# ============================================================
# 5. STAGE B: EXTRACT [CLS] EMBEDDINGS
# ============================================================
print(f"\n{'='*60}")
print("STAGE B: EXTRACTING [CLS] EMBEDDINGS")
print(f"{'='*60}")

def extract_cls_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(cls_embedding.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_embeddings), np.concatenate(all_labels)

# Non-shuffled loaders for embedding extraction
train_loader_ns = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

cls_train, y_train = extract_cls_embeddings(model, train_loader_ns, device)
cls_val, y_val = extract_cls_embeddings(model, val_loader, device)
cls_test, y_test = extract_cls_embeddings(model, test_loader, device)

print(f"[CLS] embeddings: Train={cls_train.shape}, Val={cls_val.shape}, Test={cls_test.shape}")

# ============================================================
# 6. STAGE C: MULTI-MODAL FUSION
# ============================================================
print(f"\n{'='*60}")
print("STAGE C: MULTI-MODAL FUSION")
print(f"{'='*60}")

# Get acoustic features (already in the merged df, aligned with same order)
avail_acoustic = [c for c in acoustic_cols if c in df.columns]
print(f"Acoustic features available: {avail_acoustic}")

if len(avail_acoustic) > 0:
    acou_train = train_df[avail_acoustic].values
    acou_val = val_df[avail_acoustic].values
    acou_test = test_df[avail_acoustic].values

    # Scale acoustic features
    acou_scaler = StandardScaler()
    acou_train = acou_scaler.fit_transform(acou_train)
    acou_val = acou_scaler.transform(acou_val)
    acou_test = acou_scaler.transform(acou_test)

    # Fuse: [CLS] (768) + acoustic (7)
    X_fused_train = np.hstack([cls_train, acou_train])
    X_fused_val = np.hstack([cls_val, acou_val])
    X_fused_test = np.hstack([cls_test, acou_test])

    print(f"Fused features: {X_fused_train.shape[1]} dims (768 BERT + {len(avail_acoustic)} acoustic)")

    # Train fusion classifier
    fused_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=SEED, eval_metric='logloss', verbosity=0
    )
    fused_model.fit(X_fused_train, y_train)

    # Evaluate fusion on test set
    fused_preds = fused_model.predict(X_fused_test)
    fused_probs = fused_model.predict_proba(X_fused_test)[:, 1]

    fused_acc = accuracy_score(y_test, fused_preds)
    fused_f1 = f1_score(y_test, fused_preds, average='macro')
    fused_auc = roc_auc_score(y_test, fused_probs)
    fused_prec = precision_score(y_test, fused_preds, average='macro')
    fused_rec = recall_score(y_test, fused_preds, average='macro')
    fused_cm = confusion_matrix(y_test, fused_preds)

    print(f"\nMulti-Modal Fusion Test Results:")
    print(f"  Accuracy:  {fused_acc:.3f}")
    print(f"  F1 (macro):{fused_f1:.3f}")
    print(f"  ROC-AUC:   {fused_auc:.3f}")
    print(f"  Precision: {fused_prec:.3f}")
    print(f"  Recall:    {fused_rec:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={fused_cm[0,0]}  FP={fused_cm[0,1]}")
    print(f"    FN={fused_cm[1,0]}  TP={fused_cm[1,1]}")

    # Also train acoustic-only XGBoost for comparison
    acou_only_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=SEED, eval_metric='logloss', verbosity=0
    )
    acou_only_model.fit(acou_train, y_train)
    acou_preds = acou_only_model.predict(acou_test)
    acou_probs = acou_only_model.predict_proba(acou_test)[:, 1]
    acou_acc = accuracy_score(y_test, acou_preds)
    acou_f1 = f1_score(y_test, acou_preds, average='macro')
    acou_auc = roc_auc_score(y_test, acou_probs)
else:
    print("WARNING: No acoustic features found in merged data. Skipping fusion.")
    fused_acc = fused_f1 = fused_auc = 0
    acou_acc = acou_f1 = acou_auc = 0
    fused_probs = test_probs
    acou_probs = test_probs

# Also train CLS-only XGBoost for fair comparison
cls_only_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=SEED, eval_metric='logloss', verbosity=0
)
cls_only_model.fit(cls_train, y_train)
cls_xgb_preds = cls_only_model.predict(cls_test)
cls_xgb_probs = cls_only_model.predict_proba(cls_test)[:, 1]
cls_xgb_acc = accuracy_score(y_test, cls_xgb_preds)
cls_xgb_f1 = f1_score(y_test, cls_xgb_preds, average='macro')
cls_xgb_auc = roc_auc_score(y_test, cls_xgb_probs)

# ============================================================
# 7. STAGE D: COMPARISON
# ============================================================
print(f"\n{'='*60}")
print("FINAL COMPARISON (TEST SET)")
print(f"{'='*60}")
print(f"{'Model':<35} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print(f"{'-'*55}")
print(f"{'Acoustic-only (XGBoost)':<35} {acou_acc:>6.3f} {acou_f1:>6.3f} {acou_auc:>6.3f}")
print(f"{'BERT-only (fine-tuned)':<35} {bert_acc:>6.3f} {bert_f1:>6.3f} {bert_auc:>6.3f}")
print(f"{'BERT [CLS] + XGBoost':<35} {cls_xgb_acc:>6.3f} {cls_xgb_f1:>6.3f} {cls_xgb_auc:>6.3f}")
print(f"{'MULTI-MODAL FUSION':<35} {fused_acc:>6.3f} {fused_f1:>6.3f} {fused_auc:>6.3f}")

# Load baseline XGBoost result for comparison
try:
    with open(f'{OUT_DIR}/test_results.json', 'r') as f:
        baseline_results = json.load(f)
    xgb_baseline = baseline_results.get('XGBoost (scale_pos_wt)', {})
    if xgb_baseline:
        print(f"{'XGBoost baseline (20 features)':<35} {xgb_baseline['accuracy']:>6.3f} {xgb_baseline['f1_macro']:>6.3f} {xgb_baseline['roc_auc']:>6.3f}")
except:
    pass

# ============================================================
# 8. PLOTS
# ============================================================
colors = ['#27AE60', '#2E75B6', '#8E44AD', '#C0392B', '#E67E22']

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_names = ['Acoustic\nonly', 'BERT\nfine-tuned', 'BERT [CLS]\n+ XGBoost', 'Multi-modal\nfusion']
metrics_vals = {
    'Accuracy': [acou_acc, bert_acc, cls_xgb_acc, fused_acc],
    'F1 (macro)': [acou_f1, bert_f1, cls_xgb_f1, fused_f1],
    'ROC-AUC': [acou_auc, bert_auc, cls_xgb_auc, fused_auc],
}
x = np.arange(len(model_names))
width = 0.25
for i, (metric, vals) in enumerate(metrics_vals.items()):
    bars = ax.bar(x + i*width, vals, width, label=metric, color=colors[i], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Multi-Modal Comparison (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/multimodal_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs, color in [
    ('Acoustic only', acou_probs, colors[0]),
    ('BERT fine-tuned', test_probs, colors[1]),
    ('BERT [CLS] + XGBoost', cls_xgb_probs, colors[2]),
    ('Multi-modal fusion', fused_probs, colors[3]),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Multi-Modal Comparison (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/multimodal_roc.png', dpi=150, bbox_inches='tight')
plt.close()

# Training loss curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, EPOCHS+1), train_losses, 'b-o', lw=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('BERT Training Loss')
ax1.grid(alpha=0.3)
ax2.plot(range(1, EPOCHS+1), val_f1s, 'g-o', lw=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation F1 (macro)')
ax2.set_title('BERT Validation F1')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/bert_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Save results
results = {
    'acoustic_only': {'accuracy': acou_acc, 'f1_macro': acou_f1, 'roc_auc': acou_auc},
    'bert_finetuned': {'accuracy': bert_acc, 'f1_macro': bert_f1, 'roc_auc': bert_auc,
                       'precision_macro': bert_prec, 'recall_macro': bert_rec,
                       'confusion_matrix': bert_cm.tolist()},
    'bert_cls_xgboost': {'accuracy': cls_xgb_acc, 'f1_macro': cls_xgb_f1, 'roc_auc': cls_xgb_auc},
    'multimodal_fusion': {'accuracy': fused_acc, 'f1_macro': fused_f1, 'roc_auc': fused_auc,
                          'precision_macro': fused_prec, 'recall_macro': fused_rec,
                          'confusion_matrix': fused_cm.tolist() if len(avail_acoustic) > 0 else []},
}

with open(f'{OUT_DIR}/multimodal_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT_DIR}/:")
print(f"  multimodal_comparison.png")
print(f"  multimodal_roc.png")
print(f"  bert_training_curves.png")
print(f"  multimodal_results.json")
print(f"\nModel saved to {MODEL_DIR}/bert_best.pt")
print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
