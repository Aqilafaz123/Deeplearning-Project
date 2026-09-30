import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.text_preprocessing import clean_text_id, TextVocabulary
from src.models.text_cnn import YoonKimTextCNN


from src.models.generative_reasoner import GenerativeFactChecker


class HoaxDetectorInference:
    """
    Unified Inference Engine to test new texts across all 3 models:
    - SVM + TF-IDF
    - TextCNN (Yoon Kim 2014)
    - IndoBERT (Fine-Tuned)
    Optional: Generative LLM Reasoning (Explainable AI)
    """
    def __init__(self, models_dir="saved_models"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = {0: "NON-HOAX (Berita Valid)", 1: "HOAX (Berita Bohong)"}
        
        # 1. Load SVM
        svm_path = os.path.join(models_dir, "svm_tfidf_pipeline.joblib")
        if os.path.exists(svm_path):
            self.svm = joblib.load(svm_path)
            print("[Inference] Loaded SVM + TF-IDF.")
        else:
            self.svm = None
            print(f"[Inference] Warning: {svm_path} not found.")

        # 2. Load TextCNN
        vocab_path = os.path.join(models_dir, "text_cnn_vocab.pkl")
        cnn_weights = os.path.join(models_dir, "best_text_cnn.pt")
        if os.path.exists(vocab_path) and os.path.exists(cnn_weights):
            with open(vocab_path, "rb") as f:
                self.cnn_vocab = pickle.load(f)
            self.cnn_model = YoonKimTextCNN(
                vocab_size=len(self.cnn_vocab),
                embed_dim=128,
                num_filters=100,
                filter_sizes=(3, 4, 5),
                num_classes=2
            ).to(self.device)
            self.cnn_model.load_state_dict(torch.load(cnn_weights, map_location=self.device, weights_only=True))
            self.cnn_model.eval()
            print("[Inference] Loaded TextCNN.")
        else:
            self.cnn_model = None
            print("[Inference] Warning: TextCNN weights or vocab not found.")

        # 3. Load IndoBERT
        bert_dir = os.path.join(models_dir, "best_indobert")
        if os.path.exists(bert_dir):
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self.device)
            self.bert_model.eval()
            print("[Inference] Loaded IndoBERT.")
        else:
            self.bert_model = None
            print(f"[Inference] Warning: {bert_dir} not found.")

    def predict(self, text):
        results = {}
        cleaned = clean_text_id(text)
        
        # 1. Predict SVM
        if self.svm:
            svm_pred = self.svm.predict([text])[0]
            svm_prob = self.svm.predict_proba([text])[0]
            results['SVM + TF-IDF'] = {
                'label': self.class_labels[svm_pred],
                'pred_class': int(svm_pred),
                'confidence': float(svm_prob[svm_pred]),
                'prob_non_hoax': float(svm_prob[0]),
                'prob_hoax': float(svm_prob[1])
            }

        # 2. Predict TextCNN
        if self.cnn_model:
            input_ids = self.cnn_vocab.encode(text, max_len=256)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.cnn_model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                cnn_pred = int(probs.argmax())
            results['TextCNN'] = {
                'label': self.class_labels[cnn_pred],
                'pred_class': cnn_pred,
                'confidence': float(probs[cnn_pred]),
                'prob_non_hoax': float(probs[0]),
                'prob_hoax': float(probs[1])
            }

        # 3. Predict IndoBERT
        if self.bert_model:
            inputs = self.bert_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                bert_pred = int(probs.argmax())
            results['IndoBERT'] = {
                'label': self.class_labels[bert_pred],
                'pred_class': bert_pred,
                'confidence': float(probs[bert_pred]),
                'prob_non_hoax': float(probs[0]),
                'prob_hoax': float(probs[1])
            }

        return results

    def print_prediction(self, text, explain=False, api_key=None):
        print("\n" + "="*70)
        print("                 HASIL PREDIKSI DETEKSI HOAKS                ")
        print("="*70)
        print(f"Teks Input:\n\"{text[:300]}{'...' if len(text)>300 else ''}\"\n")
        print("-"*70)
        print(f"{'Model':<20} | {'Prediksi':<25} | {'Keyakinan (Confidence)'}")
        print("-"*70)
        results = self.predict(text)
        for model_name, res in results.items():
            conf_str = f"{res['confidence']*100:.2f}% (Hoax: {res['prob_hoax']*100:.1f}%, Non-Hoax: {res['prob_non_hoax']*100:.1f}%)"
            print(f"{model_name:<20} | {res['label']:<25} | {conf_str}")
        print("="*70 + "\n")

        # Generative AI Reasoning
        if explain:
            top_model = 'IndoBERT' if 'IndoBERT' in results else list(results.keys())[0]
            label = results[top_model]['label']
            conf = results[top_model]['confidence']
            checker = GenerativeFactChecker(api_key=api_key)
            explanation = checker.explain(text, predicted_label=label, confidence=conf)
            print("="*70)
            print("      PENALARAN GENERATIVE AI (EXPLAINABLE FACT-CHECKING)      ")
            print("="*70)
            print(explanation)
            print("="*70 + "\n")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test news text on all 3 trained models")
    parser.add_argument("--text", type=str, default=None, help="News text to predict")
    parser.add_argument("--explain", action="store_true", help="Generate AI reasoning explanation")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API Key")
    args = parser.parse_args()

    detector = HoaxDetectorInference()

    if args.text:
        detector.print_prediction(args.text, explain=args.explain, api_key=args.api_key)
    else:
        sample_hoax = "Beredar kabar bahwa Presiden telah menetapkan tanggal 30 September menjadi hari libur nasional resmi."
        detector.print_prediction(sample_hoax, explain=args.explain, api_key=args.api_key)
