

"""
BERT Baseline - Evaluation Script
Usage: python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModel, BertTokenizerFast
from sklearn.metrics import (classification_report, f1_score, recall_score,
                             precision_score, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

# ── Config ────────────────────────────────────────────────────────────────────

TEST_PATH   = 'TruthCheck/src/BERT_baseline/data/berttest.csv'
MODEL_PATH  = 'TruthCheck/src/BERT_baseline/outputs/saved_weights.pt'
OUT_DIR     = 'outputs'
MAX_SEQ_LEN = 25
BATCH_SIZE  = 32

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(0.1)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.relu(self.fc1(cls_hs))
        x = self.dropout(x)
        return self.softmax(self.fc2(x))

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model...")
bert  = AutoModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TEST_PATH)
df.dropna(subset=['text'], inplace=True)
df['text'] = df['text'].astype(str)
print(f"Test set size: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokens = tokenizer(
    df['text'].tolist(),
    max_length=MAX_SEQ_LEN,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

test_seq  = torch.tensor(tokens['input_ids'])
test_mask = torch.tensor(tokens['attention_mask'])
test_y    = torch.tensor(df['label'].tolist())

# ── Run Predictions in Batches ────────────────────────────────────────────────
print("\nRunning predictions...")
all_preds = []

for i in range(0, len(test_seq), BATCH_SIZE):
    batch_seq  = test_seq[i:i+BATCH_SIZE].to(device)
    batch_mask = test_mask[i:i+BATCH_SIZE].to(device)
    with torch.no_grad():
        preds = model(batch_seq, batch_mask).detach().cpu().numpy()
    all_preds.append(preds)
    if (i // BATCH_SIZE) % 20 == 0:
        print(f"  Processed {min(i+BATCH_SIZE, len(test_seq))} / {len(test_seq)}")

all_preds = np.concatenate(all_preds, axis=0)

# ── Metrics ───────────────────────────────────────────────────────────────────
true_labels = test_y.numpy()
pred_labels = np.argmax(all_preds, axis=1)
pred_probs  = np.exp(all_preds[:, 1])  # log-prob → prob for class 1 (Real)

precision_, recall_, thresholds = precision_recall_curve(true_labels, pred_probs)

mcc       = matthews_corrcoef(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall    = recall_score(true_labels, pred_labels, zero_division=0)
f1        = f1_score(true_labels, pred_labels, average='weighted')
cm        = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print("  BERT BASELINE — TEST RESULTS")
print("="*55)
print(f"  Matthews Corr Coef  : {mcc:.4f}")
print(f"  Precision           : {precision:.4f}")
print(f"  Recall              : {recall:.4f}")
print(f"  F1 (weighted)       : {f1:.4f}")
print(f"  True Positives      : {tp}")
print(f"  True Negatives      : {tn}")
print(f"  False Positives     : {fp}")
print(f"  False Negatives     : {fn}")
print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
print(f"              Pred:Fake  Pred:Real")
print(f"  Act:Fake      {tn:>6}     {fp:>6}")
print(f"  Act:Real      {fn:>6}     {tp:>6}")
print("="*55)
print("\nPer-class report:")
print(classification_report(true_labels, pred_labels, target_names=['Fake (0)', 'Real (1)']))

# ── Save Results ──────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(recall_, precision_, marker='.', label='BERT Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — BERT Baseline')
plt.legend()
plt.tight_layout()
curve_path = os.path.join(OUT_DIR, 'precision_recall_curve.png')
plt.savefig(curve_path)
print(f"\nPrecision-Recall curve saved to {curve_path}")

df['pred_label']     = pred_labels
df['pred_prob_real'] = pred_probs
df['correct']        = (df['pred_label'] == df['label']).astype(int)
pred_csv = os.path.join(OUT_DIR, 'test_predictions.csv')
df.to_csv(pred_csv, index=False)
print(f"Predictions saved to {pred_csv}")

summary_path = os.path.join(OUT_DIR, 'eval_results.txt')
with open(summary_path, 'w') as f:
    f.write("BERT BASELINE — EVALUATION RESULTS\n")
    f.write("="*40 + "\n")
    f.write(f"Test file  : {TEST_PATH}\n")
    f.write(f"Model      : {MODEL_PATH}\n")
    f.write(f"Samples    : {len(df)}\n\n")
    f.write(f"MCC        : {mcc:.4f}\n")
    f.write(f"Precision  : {precision:.4f}\n")
    f.write(f"Recall     : {recall:.4f}\n")
    f.write(f"F1 (wtd)   : {f1:.4f}\n")
    f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n\n")
    f.write(classification_report(true_labels, pred_labels, target_names=['Fake (0)', 'Real (1)']))
print(f"Summary saved to {summary_path}")