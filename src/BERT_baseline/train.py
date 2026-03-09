
"""
BERT Baseline - Training Script
Usage: python train.py
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score, recall_score,
                             precision_score, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = 'TruthCheck/src/BERT_baseline/data/berttrain.csv'
OUT_DIR    = 'TruthCheck/src/BERT_baseline/outputs'
MAX_SEQ_LEN = 25
BATCH_SIZE  = 8
LEARNING_RATE = 3e-5
ADAM_EPSILON  = 1e-8
NUM_EPOCHS    = 3

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_PATH)
df.dropna(inplace=True)

# Support both numeric (0/1) and string ("Fake"/"Real") labels
if df['label'].dtype == object:
    df['label'] = df['label'].map({'Fake': 0, 'Real': 1, 'fake': 0, 'real': 1,
                                   'FAKE': 0, 'REAL': 1, '0': 0, '1': 1})

df = df.head(10)
print(f"Dataset size: {len(df)} | Label distribution:\n{df['label'].value_counts()}")

# ── Train / Val / Test Split ───────────────────────────────────────────────────
train_text, temp_text, train_labels, temp_labels = train_test_split(
    df['text'], df['label'], random_state=2018, test_size=0.4)

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, random_state=2018, test_size=0.5)

print(f"Train: {len(train_text)} | Val: {len(val_text)} | Test: {len(test_text)}")

# ── Tokenization ──────────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

tokens_train = tokenize(train_text)
tokens_val   = tokenize(val_text)
tokens_test  = tokenize(test_text)

# ── Tensors & DataLoaders ─────────────────────────────────────────────────────
def make_loader(tokens, labels, sampler_cls, batch_size):
    seq  = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y    = torch.tensor(labels.tolist())
    data = TensorDataset(seq, mask, y)
    return DataLoader(data, sampler=sampler_cls(data), batch_size=batch_size), seq, mask, y

train_loader, _, _, _          = make_loader(tokens_train, train_labels, RandomSampler,     BATCH_SIZE)
val_loader,   _, _, _          = make_loader(tokens_val,   val_labels,   SequentialSampler, BATCH_SIZE)
_, test_seq, test_mask, test_y = make_loader(tokens_test,  test_labels,  SequentialSampler, BATCH_SIZE)

# ── Model Architecture ────────────────────────────────────────────────────────
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert     = bert
        self.dropout  = nn.Dropout(0.1)
        self.relu     = nn.ReLU()
        self.fc1      = nn.Linear(768, 512)
        self.fc2      = nn.Linear(512, 2)
        self.softmax  = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.relu(self.fc1(cls_hs))
        x = self.dropout(x)
        return self.softmax(self.fc2(x))

bert  = AutoModel.from_pretrained('bert-base-uncased')
for param in bert.parameters():
    param.requires_grad = False

model     = BERT_Arch(bert).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)

# ── Class Weights & Loss ──────────────────────────────────────────────────────
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights   = torch.tensor(class_wts, dtype=torch.float).to(device)
criterion = nn.NLLLoss(weight=weights)

# ── Train / Eval Functions ────────────────────────────────────────────────────
def train_epoch():
    model.train()
    total_loss, all_preds = 0, []
    for step, batch in enumerate(train_loader):
        if step % 50 == 0 and step > 0:
            print(f'  Batch {step:>5} / {len(train_loader):>5}')
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss  = criterion(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        all_preds.append(preds.detach().cpu().numpy())
    return total_loss / len(train_loader), np.concatenate(all_preds, axis=0)

def eval_epoch():
    model.eval()
    total_loss, all_preds = 0, []
    for step, batch in enumerate(val_loader):
        if step % 50 == 0 and step > 0:
            print(f'  Batch {step:>5} / {len(val_loader):>5}')
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss  = criterion(preds, labels)
        total_loss += loss.item()
        all_preds.append(preds.detach().cpu().numpy())
    return total_loss / len(val_loader), np.concatenate(all_preds, axis=0)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_valid_loss = float('inf')
train_losses, valid_losses = [], []

for epoch in range(NUM_EPOCHS):
    print(f'\nEpoch {epoch+1} / {NUM_EPOCHS}')
    train_loss, _ = train_epoch()
    valid_loss, _ = eval_epoch()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'saved_weights.pt'))
        print(f'  ✓ Best model saved (val_loss={valid_loss:.4f})')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'  Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}')

# ── Save Losses & Plot ────────────────────────────────────────────────────────
joblib.dump(train_losses, os.path.join(OUT_DIR, 'train_losses.pkl'))
joblib.dump(valid_losses, os.path.join(OUT_DIR, 'val_losses.pkl'))

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig(os.path.join(OUT_DIR, 'losses.png'))
print(f'\nLoss plot saved to {OUT_DIR}\\losses.png')

# ── Test Evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'saved_weights.pt')))
model.eval()

with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

precision_, recall_, proba = precision_recall_curve(test_y, preds[:, -1])
preds_cls = np.argmax(preds, axis=1)

plt.figure()
plt.plot(recall_, precision_, marker='.', label='BERT Baseline')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
plt.savefig(os.path.join(OUT_DIR, 'precision_recall_curve.png'))

mcc       = matthews_corrcoef(test_y, preds_cls)
precision = precision_score(test_y, preds_cls)
recall    = recall_score(test_y, preds_cls)
f1        = f1_score(test_y, preds_cls, average='weighted')
cm        = confusion_matrix(test_y, preds_cls)

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Matthews Corr Coef : {mcc:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 (weighted)      : {f1:.4f}")
print(f"Confusion Matrix   :\n{cm}")
print("\nClassification Report:")
print(classification_report(test_y, preds_cls, target_names=['Fake', 'Real']))

joblib.dump({'mcc': mcc, 'precision': precision, 'recall': recall,
             'f1_weighted': f1, 'confusion_matrix': cm.tolist()},
            os.path.join(OUT_DIR, 'test_metrics.pkl'))
print(f"Metrics saved to {OUT_DIR}\\test_metrics.pkl")