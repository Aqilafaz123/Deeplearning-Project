"""
Script pengujian penuh end-to-end seluruh analisis SHAP:
1. SVM + TF-IDF (LinearExplainer)
2. TextCNN (KernelExplainer)
3. IndoBERT (PartitionExplainer / Text masker)
4. IndoBERT Waterfall Plot
"""
import os
import re
import pickle
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[1/5] Lingkungan Komputasi: {device}")

teks_uji = "Beredar kabar bahwa Presiden telah menetapkan tanggal 30 September menjadi hari libur nasional resmi."
TOP_N = 15

# =================================================================
# 1. SHAP SVM + TF-IDF
# =================================================================
print('\n' + '='*65)
print('  [2/5] SHAP ANALYSIS -- SVM + TF-IDF (LinearExplainer)')
print('='*65)

svm_pipeline = joblib.load('saved_models/svm_tfidf_pipeline.joblib')
tfidf_step = svm_pipeline.named_steps['tfidf']
calibrated = svm_pipeline.named_steps['clf']
base_svc = calibrated.calibrated_classifiers_[0].estimator
feature_names = tfidf_step.get_feature_names_out()

n_feats = len(feature_names)
background_svm = np.zeros((1, n_feats))
svm_explainer = shap.LinearExplainer(base_svc, background_svm)

X_vec = tfidf_step.transform([teks_uji])
shap_vals_svm = svm_explainer.shap_values(X_vec)
sv_hoax = shap_vals_svm[1][0] if isinstance(shap_vals_svm, list) else shap_vals_svm[0]

X_dense = X_vec.toarray()[0]
active_idx = np.where(X_dense > 0)[0]

if len(active_idx) > 0:
    active_sv = sv_hoax[active_idx]
    active_tokens = feature_names[active_idx]
    order = np.argsort(np.abs(active_sv))[::-1][:TOP_N]
    sorted_tokens = active_tokens[order]
    sorted_sv = active_sv[order]

    colors = ['#e63946' if v > 0 else '#2a9d8f' for v in sorted_sv]
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_tokens) * 0.4)), dpi=120)
    ax.barh(range(len(sorted_tokens)), sorted_sv, color=colors)
    ax.set_yticks(range(len(sorted_tokens)))
    ax.set_yticklabels(sorted_tokens, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value (merah -> HOAX | hijau -> NON-HOAX)')
    ax.set_title('SHAP SVM + TF-IDF - Token Paling Berpengaruh', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/shap_svm.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Token pendorong HOAX    : {[t for t, v in zip(sorted_tokens, sorted_sv) if v > 0][:5]}')
    print(f'  Token pendorong NON-HOAX: {[t for t, v in zip(sorted_tokens, sorted_sv) if v < 0][:5]}')
    print('  [OK] Plot SVM disimpan ke results/shap_svm.png')

# =================================================================
# 2. SHAP TextCNN
# =================================================================
print('\n' + '='*65)
print('  [3/5] SHAP ANALYSIS -- TextCNN (KernelExplainer)')
print('='*65)

with open('saved_models/text_cnn_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.max(torch.relu(conv(emb)), dim=2)[0] for conv in self.convs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

cnn_model = TextCNNModel(vocab_size=len(vocab.word2idx)).to(device)
cnn_model.load_state_dict(torch.load('saved_models/best_text_cnn.pt', map_location=device))
cnn_model.eval()

def tokenize_simple(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return [t for t in text.split() if t]

def words_to_ids(words, vocab_obj, max_len=256):
    UNK = vocab_obj.word2idx.get('<UNK>', 1)
    ids = [vocab_obj.word2idx.get(w, UNK) for w in words]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids

def predict_cnn_texts(text_list):
    cnn_model.eval()
    probs_list = []
    for txt in text_list:
        words = tokenize_simple(txt)
        ids = words_to_ids(words, vocab)
        t = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = cnn_model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_list.append(probs)
    return np.array(probs_list)

words_uji = tokenize_simple(teks_uji)
print(f'  Jumlah kata dalam teks uji: {len(words_uji)}')

def masked_predict(masks):
    texts = []
    for m in masks:
        masked_words = [w if m[j] else '' for j, w in enumerate(words_uji)]
        texts.append(' '.join(masked_words))
    return predict_cnn_texts(texts)

baseline_mask = np.ones((1, len(words_uji)), dtype=bool)
cnn_explainer = shap.KernelExplainer(masked_predict, baseline_mask)

print('  Menghitung SHAP values TextCNN (nsamples=100)...')
shap_vals_cnn = cnn_explainer.shap_values(
    np.ones((1, len(words_uji)), dtype=bool),
    nsamples=100, silent=True
)

if isinstance(shap_vals_cnn, list):
    sv_cnn = shap_vals_cnn[1][0]
elif isinstance(shap_vals_cnn, np.ndarray):
    if shap_vals_cnn.ndim == 3:
        sv_cnn = shap_vals_cnn[0, :, 1]
    elif shap_vals_cnn.ndim == 2:
        sv_cnn = shap_vals_cnn[0]
    else:
        sv_cnn = shap_vals_cnn
else:
    sv_cnn = np.array(shap_vals_cnn)

sv_cnn = np.squeeze(np.asarray(sv_cnn, dtype=float))

order_cnn = np.argsort(np.abs(sv_cnn))[::-1][:TOP_N]
sorted_words = [words_uji[i] for i in order_cnn]
sorted_sv_cnn = sv_cnn[order_cnn]

colors_cnn = ['#e63946' if v > 0 else '#2a9d8f' for v in sorted_sv_cnn]
fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_words) * 0.4)), dpi=120)
ax.barh(range(len(sorted_words)), sorted_sv_cnn, color=colors_cnn)
ax.set_yticks(range(len(sorted_words)))
ax.set_yticklabels(sorted_words, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('SHAP Value (merah -> HOAX | hijau -> NON-HOAX)')
ax.set_title('SHAP TextCNN - Kata Paling Berpengaruh', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap_textcnn.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'  Kata pendorong HOAX    : {[w for w, v in zip(sorted_words, sorted_sv_cnn) if v > 0][:5]}')
print(f'  Kata pendorong NON-HOAX: {[w for w, v in zip(sorted_words, sorted_sv_cnn) if v < 0][:5]}')
print('  [OK] Plot TextCNN disimpan ke results/shap_textcnn.png')

# =================================================================
# 3. SHAP IndoBERT
# =================================================================
print('\n' + '='*65)
print('  [4/5] SHAP ANALYSIS -- IndoBERT (PartitionExplainer)')
print('='*65)

tokenizer = AutoTokenizer.from_pretrained('saved_models/best_indobert')
bert_model = AutoModelForSequenceClassification.from_pretrained('saved_models/best_indobert').to(device)
bert_model.eval()

def predict_bert_texts(texts):
    bert_model.eval()
    all_probs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)

bert_masker = shap.maskers.Text(tokenizer)
bert_explainer = shap.Explainer(
    predict_bert_texts,
    masker=bert_masker,
    output_names=['NON-HOAX', 'HOAX']
)

print('  Menghitung SHAP values IndoBERT (max_evals=200)...')
shap_values_bert = bert_explainer([teks_uji], max_evals=200, batch_size=1)

sv_bert = shap_values_bert[0, :, 1].values
tokens_bert = shap_values_bert[0, :, 1].data

order_bert = np.argsort(np.abs(sv_bert))[::-1][:TOP_N]
sorted_tok = [tokens_bert[i] for i in order_bert]
sorted_sv_b = sv_bert[order_bert]

colors_bert = ['#e63946' if v > 0 else '#2a9d8f' for v in sorted_sv_b]
fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_tok) * 0.4)), dpi=120)
ax.barh(range(len(sorted_tok)), sorted_sv_b, color=colors_bert)
ax.set_yticks(range(len(sorted_tok)))
ax.set_yticklabels(sorted_tok, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('SHAP Value (merah -> HOAX | hijau -> NON-HOAX)')
ax.set_title('SHAP IndoBERT - Token Paling Berpengaruh', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap_indobert.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'  Token pendorong HOAX    : {[t for t, v in zip(sorted_tok, sorted_sv_b) if v > 0][:5]}')
print(f'  Token pendorong NON-HOAX: {[t for t, v in zip(sorted_tok, sorted_sv_b) if v < 0][:5]}')
print('  [OK] Plot IndoBERT disimpan ke results/shap_indobert.png')

# =================================================================
# 4. SHAP Waterfall Chart
# =================================================================
print('\n' + '='*65)
print('  [5/5] SHAP Waterfall Chart -- IndoBERT')
print('='*65)

fig = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_bert[0, :, 1], show=False)
plt.tight_layout()
plt.savefig('results/shap_indobert_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] Plot Waterfall disimpan ke results/shap_indobert_waterfall.png')

print('\n' + '='*65)
print('  SEMUA ANALISIS SHAP SELESAI DENGAN SUKSES! (100% COMPLETE)')
print('='*65)
