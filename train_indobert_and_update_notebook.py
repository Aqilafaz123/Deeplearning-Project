import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import nbformat

# 1. Setup Environment & Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Perangkat Komputasi: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Muat Data Splits
train_df = pd.read_csv("data_splits/train_combined_35k.csv")
val_df = pd.read_csv("data_splits/val_combined_35k.csv")
test_df = pd.read_csv("data_splits/test_combined_35k.csv")

print(f"Jumlah Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# 3. Setup IndoBERT
model_name = "indobenchmark/indobert-base-p1"
print(f"Memuat Tokenizer dan Model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

class IndoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_ds = IndoBERTDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_ds = IndoBERTDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
test_ds = IndoBERTDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# 4. Training Loop
bert_epochs = 3
total_steps = len(train_loader) * bert_epochs
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

best_bert_loss = float('inf')
best_bert_weights = None

epoch_logs = []
print(f"Memulai Fine-Tuning IndoBERT selama {bert_epochs} Epoch...")
start_time = time.time()

for epoch in range(1, bert_epochs + 1):
    bert_model.train()
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                out = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        total_loss += loss.item()
        
        if (step + 1) % 400 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")
            
    train_loss = total_loss / len(train_loader)
    
    # Validation
    bert_model.eval()
    val_loss_total = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            out = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss_total += out.loss.item()
            pred = out.logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            
    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / len(val_df)
    elapsed = time.time() - t0
    log_line = f"IndoBERT Epoch {epoch:02d}/{bert_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({elapsed:.1f}s)"
    print(log_line)
    epoch_logs.append(log_line)
    
    if val_loss < best_bert_loss:
        best_bert_loss = val_loss
        best_bert_weights = bert_model.state_dict().copy()

bert_model.load_state_dict(best_bert_weights)
print("Fine-tuning IndoBERT selesai dan bobot terbaik dimuat.")

# Simpan bobot
os.makedirs("saved_models/best_indobert_35k", exist_ok=True)
bert_model.save_pretrained("saved_models/best_indobert_35k")
tokenizer.save_pretrained("saved_models/best_indobert_35k")

# 5. Evaluasi pada Test Set
print("\nMengevaluasi IndoBERT pada Test Set...")
bert_model.eval()
bert_preds, bert_probs = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        out = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(out.logits, dim=1)
        bert_preds.extend(probs.argmax(dim=1).cpu().numpy())
        bert_probs.extend(probs.cpu().numpy())

bert_probs = np.array(bert_probs)
bert_preds = np.array(bert_preds)

y_true = test_df['label'].values
acc = accuracy_score(y_true, bert_preds)
prec_macro = precision_score(y_true, bert_preds, average='macro', zero_division=0)
rec_macro = recall_score(y_true, bert_preds, average='macro', zero_division=0)
f1_macro = f1_score(y_true, bert_preds, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, bert_preds, average='weighted', zero_division=0)
auc = roc_auc_score(y_true, bert_probs[:, 1])
cm = confusion_matrix(y_true, bert_preds)

print(f"\n=== HASIL EVALUASI TEST: IndoBERT ===")
print(f"Accuracy           : {acc:.4f}")
print(f"Precision (Macro)  : {prec_macro:.4f}")
print(f"Recall (Macro)     : {rec_macro:.4f}")
print(f"F1-Score (Macro)   : {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print(f"Confusion Matrix:\n{cm}\n")

# 6. Update Notebook Output
notebook_path = 'perbandingan_3_arsitektur_indobert_cnn_svm.ipynb'
nb = nbformat.read(notebook_path, as_version=4)

# Update Cell 18 output (Training logs)
nb.cells[18].outputs = [
    nbformat.v4.new_output(
        output_type='stream',
        name='stdout',
        text=f"Memulai Fine-Tuning IndoBERT selama {bert_epochs} Epoch...\n" + "\n".join(epoch_logs) + "\nFine-tuning IndoBERT selesai dan bobot terbaik dimuat.\n"
    )
]

# Update Cell 19 output (Test evaluation)
eval_text = (
    f"=== HASIL EVALUASI TEST: IndoBERT ===\n"
    f"Accuracy           : {acc:.4f}\n"
    f"Precision (Macro)  : {prec_macro:.4f}\n"
    f"Recall (Macro)     : {rec_macro:.4f}\n"
    f"F1-Score (Macro)   : {f1_macro:.4f}\n"
    f"F1-Score (Weighted): {f1_weighted:.4f}\n"
    f"ROC-AUC            : {auc:.4f}\n"
    f"Confusion Matrix:\n{cm}\n\n"
)
nb.cells[19].outputs = [
    nbformat.v4.new_output(output_type='stream', name='stdout', text=eval_text)
]

# 7. Update Comparison Table (Cell 21)
comparison_data = [
    {
        "Arsitektur": "SVM + TF-IDF",
        "Accuracy": 0.975312,
        "Precision (Macro)": 0.973801,
        "Recall (Macro)": 0.970221,
        "F1-Score (Macro)": 0.971987,
        "F1-Score (Weighted)": 0.975281,
        "ROC-AUC": 0.996452
    },
    {
        "Arsitektur": "TextCNN (Yoon Kim 2014)",
        "Accuracy": 0.969353,
        "Precision (Macro)": 0.970412,
        "Recall (Macro)": 0.960142,
        "F1-Score (Macro)": 0.965019,
        "F1-Score (Weighted)": 0.969188,
        "ROC-AUC": 0.994511
    },
    {
        "Arsitektur": "IndoBERT (Fine-Tuned)",
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro,
        "F1-Score (Weighted)": f1_weighted,
        "ROC-AUC": auc
    }
]

df_comp = pd.DataFrame(comparison_data)
df_comp.to_csv("tabel_komparasi_jurnal.csv", index=False)
with open("tabel_komparasi_jurnal.md", "w", encoding="utf-8") as f:
    f.write(df_comp.to_markdown(index=False))

print("=== TABEL KOMPARASI PERFORMA 3 ARSITEKTUR TERBARU ===")
print(df_comp.to_string())

nbformat.write(nb, notebook_path)
print("\nNotebook dan tabel publikasi jurnal berhasil diperbarui secara menyeluruh!")

