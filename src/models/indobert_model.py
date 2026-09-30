import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm


class IndoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class IndoBERTTrainer:
    def __init__(self, model_name="indobenchmark/indobert-base-p1", num_classes=2, lr=2e-5, weight_decay=0.01, device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_name = model_name
        print(f"[IndoBERT] Initializing tokenizer and model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model.to(self.device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training IndoBERT", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
            scheduler.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, accuracy, np.array(all_preds), np.array(all_probs)

    def fit(self, train_loader, val_loader, epochs=3, warmup_ratio=0.1, save_dir="saved_models/best_indobert"):
        os.makedirs(save_dir, exist_ok=True)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Optimizer with weight decay fix
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        train_losses, val_losses, val_accs = [], [], []
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"IndoBERT Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"  -> Best IndoBERT checkpoint saved to {save_dir} (Val Loss: {val_loss:.4f})")
                
        # Reload best model
        print(f"[IndoBERT] Loading best model checkpoint from {save_dir}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        self.model.to(self.device)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
