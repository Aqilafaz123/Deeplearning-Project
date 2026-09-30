import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import evaluate_predictions, print_metrics_summary, plot_and_save_confusion_matrix
from src.models.text_cnn import YoonKimTextCNN


def evaluate_custom_test_file(file_path, text_col=None, label_col=None, models_dir="saved_models", output_dir="results/custom_test"):
    """
    Evaluates a user-provided independent test dataset on all 3 trained models.
    Supports CSV and Excel (.xlsx, .xls) files.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[CustomTest] Membaca file data uji eksternal: {file_path}")

    # 1. Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Format file harus berupa .csv atau .xlsx / .xls")

    # Detect text and label columns if not provided
    if not text_col:
        candidates = ['text', 'summarized', 'cleaned', 'narasi', 'berita', 'Clean Narasi', 'text_new']
        for c in candidates:
            if c in df.columns:
                text_col = c
                break
        if not text_col:
            raise KeyError(f"Kolom teks tidak ditemukan otomatis. Kolom yang ada: {list(df.columns)}. Gunakan flag --text_col nama_kolom")

    has_labels = False
    if not label_col:
        candidates = ['label', 'hoax', 'kategori', 'target', 'class']
        for c in candidates:
            if c in df.columns:
                label_col = c
                has_labels = True
                break
    else:
        has_labels = label_col in df.columns

    # Clean data
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist() if has_labels else None

    print(f"[CustomTest] Total sampel data uji yang dimuat: {len(texts)}")
    print(f"[CustomTest] Kolom teks: '{text_col}', Kolom label: '{label_col if has_labels else 'Tidak Ada Label (Hanya Prediksi)'}'")
    if has_labels:
        print(f"[CustomTest] Distribusi label data uji: {dict(pd.Series(labels).value_counts())}")

    summary_list = []

    # ==========================
    # 1. EVALUATE SVM
    # ==========================
    svm_path = os.path.join(models_dir, "svm_tfidf_pipeline.joblib")
    if os.path.exists(svm_path):
        print("\n>>> Menjalankan Uji Model 1: SVM + TF-IDF...")
        svm_model = joblib.load(svm_path)
        svm_preds = svm_model.predict(texts)
        svm_probs = svm_model.predict_proba(texts)
        df['pred_svm'] = svm_preds
        df['prob_svm_hoax'] = svm_probs[:, 1]
        
        if has_labels:
            m = evaluate_predictions(labels, svm_preds, svm_probs, model_name="SVM (Custom Test)")
            print_metrics_summary(m)
            plot_and_save_confusion_matrix(m['confusion_matrix'], title="Confusion Matrix SVM (Custom Test)", 
                                           save_path=os.path.join(output_dir, "cm_svm.png"))
            summary_list.append({"Model": "SVM + TF-IDF", "Accuracy": m['accuracy'], "F1-Macro": m['f1_macro'], "ROC-AUC": m.get('roc_auc')})

    # ==========================
    # 2. EVALUATE TextCNN
    # ==========================
    vocab_path = os.path.join(models_dir, "text_cnn_vocab.pkl")
    cnn_path = os.path.join(models_dir, "best_text_cnn.pt")
    if os.path.exists(vocab_path) and os.path.exists(cnn_path):
        print("\n>>> Menjalankan Uji Model 2: TextCNN (Yoon Kim 2014)...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        cnn_model = YoonKimTextCNN(vocab_size=len(vocab), embed_dim=128, num_filters=100, filter_sizes=(3, 4, 5), num_classes=2).to(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn_model.eval()

        cnn_preds, cnn_probs = [], []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = [vocab.encode(t, max_len=256) for t in batch_texts]
            tensor_x = torch.tensor(encoded, dtype=torch.long).to(device)
            with torch.no_grad():
                out = cnn_model(tensor_x)
                probs = F.softmax(out, dim=1).cpu().numpy()
                cnn_preds.extend(probs.argmax(axis=1))
                cnn_probs.extend(probs)

        df['pred_textcnn'] = cnn_preds
        df['prob_textcnn_hoax'] = np.array(cnn_probs)[:, 1]

        if has_labels:
            m = evaluate_predictions(labels, np.array(cnn_preds), np.array(cnn_probs), model_name="TextCNN (Custom Test)")
            print_metrics_summary(m)
            plot_and_save_confusion_matrix(m['confusion_matrix'], title="Confusion Matrix TextCNN (Custom Test)", 
                                           save_path=os.path.join(output_dir, "cm_textcnn.png"))
            summary_list.append({"Model": "TextCNN", "Accuracy": m['accuracy'], "F1-Macro": m['f1_macro'], "ROC-AUC": m.get('roc_auc')})

    # ==========================
    # 3. EVALUATE IndoBERT
    # ==========================
    bert_dir = os.path.join(models_dir, "best_indobert")
    if os.path.exists(bert_dir):
        print("\n>>> Menjalankan Uji Model 3: IndoBERT Fine-Tuned...")
        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
        bert_model.eval()

        bert_preds, bert_probs = [], []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                out = bert_model(**enc)
                probs = F.softmax(out.logits, dim=1).cpu().numpy()
                bert_preds.extend(probs.argmax(axis=1))
                bert_probs.extend(probs)

        df['pred_indobert'] = bert_preds
        df['prob_indobert_hoax'] = np.array(bert_probs)[:, 1]

        if has_labels:
            m = evaluate_predictions(labels, np.array(bert_preds), np.array(bert_probs), model_name="IndoBERT (Custom Test)")
            print_metrics_summary(m)
            plot_and_save_confusion_matrix(m['confusion_matrix'], title="Confusion Matrix IndoBERT (Custom Test)", 
                                           save_path=os.path.join(output_dir, "cm_indobert.png"))
            summary_list.append({"Model": "IndoBERT", "Accuracy": m['accuracy'], "F1-Macro": m['f1_macro'], "ROC-AUC": m.get('roc_auc')})

    # Save output predictions
    output_predictions_path = os.path.join(output_dir, "hasil_prediksi_custom_test.csv")
    df.to_csv(output_predictions_path, index=False)
    print(f"\n[CustomTest] File hasil prediksi lengkap tersimpan di: {output_predictions_path}")

    if has_labels and summary_list:
        df_summary = pd.DataFrame(summary_list)
        summary_csv = os.path.join(output_dir, "tabel_komparasi_custom_test.csv")
        df_summary.to_csv(summary_csv, index=False)
        print("\n=======================================================")
        print("          HASIL KOMPARASI DATA TEST SENDIRI            ")
        print("=======================================================")
        print(df_summary.to_string(index=False))
        print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate custom test dataset on trained models")
    parser.add_argument("--file", type=str, required=True, help="Path to your test dataset (.csv or .xlsx)")
    parser.add_argument("--text_col", type=str, default=None, help="Text column name")
    parser.add_argument("--label_col", type=str, default=None, help="Label column name (0 for Non-Hoax, 1 for Hoax)")
    parser.add_argument("--output_dir", type=str, default="results/custom_test")
    args = parser.parse_args()

    evaluate_custom_test_file(
        file_path=args.file,
        text_col=args.text_col,
        label_col=args.label_col,
        output_dir=args.output_dir
    )
