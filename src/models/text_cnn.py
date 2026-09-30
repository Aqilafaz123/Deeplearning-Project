import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        input_ids = self.vocab.encode(text, max_len=self.max_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class YoonKimTextCNN(nn.Module):
    """
    TextCNN Architecture following Yoon Kim (2014) 'Convolutional Neural Networks for Sentence Classification'.
    Uses parallel 1D convolutions with multiple kernel sizes (3, 4, 5) followed by max-over-time pooling.
    """
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1) # (batch_size, embed_dim, seq_len)
        
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded)) # (batch_size, num_filters, seq_len - k + 1)
            pooled = torch.max(conv_out, dim=2)[0] # (batch_size, num_filters)
            pooled_outputs.append(pooled)
            
        cat = torch.cat(pooled_outputs, dim=1) # (batch_size, num_filters * len(filter_sizes))
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits


class TextCNNTrainer:
    def __init__(self, model, device=None, lr=1e-3, weight_decay=1e-4):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
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
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, accuracy, np.array(all_preds), np.array(all_probs)

    def fit(self, train_loader, val_loader, epochs=10, patience=3, save_path="saved_models/best_text_cnn.pt"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[TextCNN] Training on device: {self.device}")
        
        train_losses, val_losses, val_accs = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Best model saved to {save_path} (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[TextCNN] Early stopping triggered after {epoch} epochs.")
                    break
                    
        # Load best model weights
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
            print(f"[TextCNN] Loaded best model weights from {save_path}")
            
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
