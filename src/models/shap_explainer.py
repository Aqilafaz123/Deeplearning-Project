"""
shap_explainer.py
=================
Explainable AI (XAI) menggunakan SHAP untuk 3 model:
  - SVM + TF-IDF     → shap.LinearExplainer  (cepat & akurat)
  - TextCNN           → shap.DeepExplainer    (berbasis gradien PyTorch)
  - IndoBERT          → shap.Explainer (Partition/Text) via transformers-pipeline

Penggunaan:
    from src.models.shap_explainer import SHAPExplainer
    xai = SHAPExplainer(detector)               # detector = HoaxDetectorInference
    xai.explain_svm(text, top_n=15)
    xai.explain_textcnn(text, background_texts)
    xai.explain_indobert(text)
    xai.explain_all(text, background_texts)     # semua sekaligus
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend aman untuk server
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Helper: cek & import shap
# ─────────────────────────────────────────────
def _require_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError(
            "Paket 'shap' belum terinstall.\n"
            "Jalankan: pip install shap"
        )


# ─────────────────────────────────────────────
# 1.  SVM SHAP (LinearExplainer)
# ─────────────────────────────────────────────
class SVMSHAPExplainer:
    """
    Menggunakan shap.LinearExplainer pada model SVM+TF-IDF.
    LinearExplainer cocok karena SVM adalah model linear di ruang TF-IDF.
    Untuk CalibratedClassifierCV kita extrak base LinearSVC-nya.
    """

    def __init__(self, svm_pipeline):
        shap = _require_shap()
        self.shap = shap

        # Ambil tahap TF-IDF dan classifier dari pipeline sklearn
        self.tfidf = svm_pipeline.named_steps["tfidf"]
        calibrated = svm_pipeline.named_steps["clf"]

        # CalibratedClassifierCV → ambil salah satu calibrators[0].estimator
        if hasattr(calibrated, "calibrated_classifiers_"):
            base_clf = calibrated.calibrated_classifiers_[0].estimator
        else:
            base_clf = calibrated

        # LinearExplainer butuh coef_ dan intercept_
        if not hasattr(base_clf, "coef_"):
            raise ValueError(
                "Base classifier tidak memiliki coef_. "
                "Pastikan menggunakan LinearSVC."
            )

        self.base_clf = base_clf
        # Masking matrix: shap value dihitung di ruang TF-IDF
        # Background = zero vector (makna: tidak ada kata)
        n_features = len(self.tfidf.get_feature_names_out())
        background = np.zeros((1, n_features))
        self.explainer = shap.LinearExplainer(base_clf, background)
        self.feature_names = self.tfidf.get_feature_names_out()

    def explain(self, text: str, top_n: int = 20, plot: bool = True,
                save_path: str = None):
        """
        Kembalikan SHAP values untuk `text` dan (opsional) tampilkan plot.

        Returns:
            dict: {
                'tokens': list[str],
                'shap_values': np.ndarray shape (n_features,) → nilai kelas HOAX (1)
                'top_positive': list[(token, shap_val)],   # mendorong → HOAX
                'top_negative': list[(token, shap_val)],   # mendorong → NON-HOAX
            }
        """
        import scipy.sparse

        X_vec = self.tfidf.transform([text])  # sparse (1, n_features)
        shap_vals = self.explainer.shap_values(X_vec)

        # shap_values bisa list[class0, class1] atau array tunggal
        if isinstance(shap_vals, list):
            # ambil SHAP untuk kelas HOAX (index 1)
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        # Ambil fitur yang non-zero di teks ini agar lebih relevan
        X_dense = X_vec.toarray()[0]
        active_mask = X_dense > 0
        active_idx = np.where(active_mask)[0]

        if len(active_idx) == 0:
            return {"tokens": [], "shap_values": sv,
                    "top_positive": [], "top_negative": []}

        active_sv = sv[active_idx]
        active_tokens = self.feature_names[active_idx]

        # Urutkan
        order = np.argsort(np.abs(active_sv))[::-1][:top_n]
        sorted_tokens = active_tokens[order]
        sorted_sv = active_sv[order]

        top_positive = [(t, float(v)) for t, v in zip(sorted_tokens, sorted_sv) if v > 0]
        top_negative = [(t, float(v)) for t, v in zip(sorted_tokens, sorted_sv) if v < 0]

        if plot:
            self._bar_plot(sorted_tokens, sorted_sv,
                           title="SHAP SVM – Token Paling Berpengaruh (Kelas HOAX)",
                           save_path=save_path)

        return {
            "tokens": sorted_tokens.tolist(),
            "shap_values": sorted_sv,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    def _bar_plot(self, tokens, values, title="SHAP Values", save_path=None):
        colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]
        fig, ax = plt.subplots(figsize=(10, max(4, len(tokens) * 0.4)))
        bars = ax.barh(range(len(tokens)), values, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (positif = mengarah HOAX, negatif = NON-HOAX)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP-SVM] Plot disimpan ke: {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────
# 2.  TextCNN SHAP (KernelExplainer / sampling)
# ─────────────────────────────────────────────
class TextCNNSHAPExplainer:
    """
    Menggunakan shap.KernelExplainer berbasis token-masking untuk TextCNN.
    DeepExplainer tidak bekerja baik dengan Embedding → kita pakai
    KernelExplainer berbasis kata (word-level perturbation).

    background_texts: list[str] – sampel teks representatif dari data train
                                   (cukup 50-100 teks untuk kecepatan)
    """

    def __init__(self, cnn_model, cnn_vocab, background_texts: list,
                 device=None, max_len: int = 256):
        import torch
        import torch.nn.functional as F
        shap = _require_shap()

        self.shap = shap
        self.cnn_model = cnn_model
        self.cnn_vocab = cnn_vocab
        self.max_len = max_len
        self.device = device or torch.device("cpu")
        self.torch = torch
        self.F = F
        self.cnn_model.eval()

        # Tokenisasi teks menjadi list kata untuk word-level SHAP
        self._background_texts = background_texts

    def _text_to_token_list(self, text: str):
        """Tokenisasi sederhana berbasis spasi setelah cleaning."""
        from src.text_preprocessing import clean_text_id
        return clean_text_id(text).split()

    def _tokens_to_ids(self, tokens):
        ids = [self.cnn_vocab.word2idx.get(t, self.cnn_vocab.word2idx.get("<UNK>", 1))
               for t in tokens]
        # Pad / truncate
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def _predict_fn(self, masked_texts):
        """
        Fungsi prediksi untuk KernelExplainer.
        Input: list[str] → teks hasil masking
        Output: np.ndarray shape (n, 2) probabilitas [non-hoax, hoax]
        """
        results = []
        for text in masked_texts:
            tokens = self._text_to_token_list(text)
            ids = self._tokens_to_ids(tokens)
            t = self.torch.tensor([ids], dtype=self.torch.long).to(self.device)
            with self.torch.no_grad():
                logits = self.cnn_model(t)
                probs = self.F.softmax(logits, dim=1).cpu().numpy()[0]
            results.append(probs)
        return np.array(results)

    def explain(self, text: str, top_n: int = 20, nsamples: int = 200,
                plot: bool = True, save_path: str = None):
        """
        Returns:
            dict: {
                'words': list[str],
                'shap_values': np.ndarray,   # shap untuk kelas HOAX
                'top_positive': list[(word, val)],
                'top_negative': list[(word, val)],
            }
        """
        words = self._text_to_token_list(text)
        if not words:
            return {"words": [], "shap_values": np.array([]),
                    "top_positive": [], "top_negative": []}

        # Masker: ganti kata yang di-"off" dengan string kosong (dihapus)
        def masker(mask, x):
            """mask: bool array, x: single text → return masked texts"""
            tokens = x.split()
            results = []
            for m in mask:
                masked = " ".join(t if m[i] else "" for i, t in enumerate(tokens))
                results.append(masked)
            return np.array(results)

        # KernelExplainer dengan background = teks kosong
        background_data = np.array([" ".join(words)])
        explainer = self.shap.KernelExplainer(
            lambda masks: self._predict_fn(
                [" ".join(w if masks[i][j] else "" for j, w in enumerate(words))
                 for i in range(len(masks))]
            ),
            data=np.ones((1, len(words)), dtype=bool)  # baseline semua kata ada
        )

        shap_vals = explainer.shap_values(
            np.ones((1, len(words)), dtype=bool),
            nsamples=nsamples,
            silent=True
        )

        # shap_vals: ambil kelas 1 (HOAX) dengan aman untuk format list maupun ndarray 3D
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            elif shap_vals.ndim == 2:
                sv = shap_vals[0]
            else:
                sv = shap_vals
        else:
            sv = np.array(shap_vals)
        sv = np.squeeze(np.asarray(sv, dtype=float))

        order = np.argsort(np.abs(sv))[::-1][:top_n]
        sorted_words = [words[i] for i in order]
        sorted_sv = sv[order]

        top_positive = [(w, float(v)) for w, v in zip(sorted_words, sorted_sv) if v > 0]
        top_negative = [(w, float(v)) for w, v in zip(sorted_words, sorted_sv) if v < 0]

        if plot:
            self._bar_plot(sorted_words, sorted_sv,
                           title="SHAP TextCNN – Kata Paling Berpengaruh (Kelas HOAX)",
                           save_path=save_path)

        return {
            "words": sorted_words,
            "shap_values": sorted_sv,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    def _bar_plot(self, words, values, title="SHAP Values", save_path=None):
        colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]
        fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.4)))
        ax.barh(range(len(words)), values, color=colors)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (positif = mengarah HOAX, negatif = NON-HOAX)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP-TextCNN] Plot disimpan ke: {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────
# 3.  IndoBERT SHAP (Partition Explainer / Text)
# ─────────────────────────────────────────────
class IndoBERTSHAPExplainer:
    """
    Menggunakan shap.Explainer dengan tokenizer transformers.
    Secara otomatis menggunakan PartitionExplainer (Shapley partition)
    yang dirancang untuk NLP berbasis token.
    """

    def __init__(self, bert_model, bert_tokenizer, device=None):
        import torch
        import torch.nn.functional as F
        shap = _require_shap()

        self.shap = shap
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.device = device or torch.device("cpu")
        self.torch = torch
        self.F = F
        self.bert_model.eval()

        # Bungkus model menjadi pipeline function untuk shap.Explainer
        self._setup_explainer()

    def _setup_explainer(self):
        """Buat shap.Explainer berbasis transformers pipeline."""
        import torch
        import torch.nn.functional as F

        def predict_fn(texts):
            all_probs = []
            for text in texts:
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
            return np.array(all_probs)

        self.predict_fn = predict_fn
        # shap.Explainer dengan masker teks otomatis (BERT WordPiece aware)
        self.masker = self.shap.maskers.Text(self.bert_tokenizer)
        self.explainer = self.shap.Explainer(
            predict_fn,
            masker=self.masker,
            output_names=["NON-HOAX", "HOAX"]
        )

    def explain(self, text: str, top_n: int = 20, plot: bool = True,
                save_path: str = None, max_evals: int = 500):
        """
        Returns:
            dict: {
                'tokens': list[str],
                'shap_values': np.ndarray,   # shap untuk kelas HOAX
                'top_positive': list[(token, val)],
                'top_negative': list[(token, val)],
                'shap_obj': shap.Explanation   # bisa dipakai untuk waterfall/text plot
            }
        """
        shap_values = self.explainer(
            [text],
            max_evals=max_evals,
            batch_size=1
        )

        # shap_values.shape: (1, n_tokens, 2)
        # Ambil kelas HOAX (index 1)
        sv_hoax = shap_values[0, :, 1].values        # (n_tokens,)
        tokens = shap_values[0, :, 1].data           # token strings

        order = np.argsort(np.abs(sv_hoax))[::-1][:top_n]
        sorted_tokens = [tokens[i] for i in order]
        sorted_sv = sv_hoax[order]

        top_positive = [(t, float(v)) for t, v in zip(sorted_tokens, sorted_sv) if v > 0]
        top_negative = [(t, float(v)) for t, v in zip(sorted_tokens, sorted_sv) if v < 0]

        if plot:
            self._bar_plot(sorted_tokens, sorted_sv,
                           title="SHAP IndoBERT – Token Paling Berpengaruh (Kelas HOAX)",
                           save_path=save_path)

        return {
            "tokens": sorted_tokens,
            "shap_values": sorted_sv,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "shap_obj": shap_values,
        }

    def plot_text(self, shap_obj, class_idx: int = 1):
        """
        Tampilkan SHAP text plot (highlight kata di teks asli).
        Hanya bekerja di Jupyter Notebook / HTML.
        """
        self.shap.plots.text(shap_obj[0, :, class_idx])

    def plot_waterfall(self, shap_obj, class_idx: int = 1, save_path: str = None):
        """Waterfall chart untuk satu prediksi."""
        fig = plt.figure()
        self.shap.plots.waterfall(shap_obj[0, :, class_idx], show=False)
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP-IndoBERT] Waterfall plot disimpan ke: {save_path}")
        plt.close(fig)

    def _bar_plot(self, tokens, values, title="SHAP Values", save_path=None):
        colors = ["#e63946" if v > 0 else "#2a9d8f" for v in values]
        fig, ax = plt.subplots(figsize=(10, max(4, len(tokens) * 0.4)))
        ax.barh(range(len(tokens)), values, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (positif = mengarah HOAX, negatif = NON-HOAX)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP-IndoBERT] Plot disimpan ke: {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────
# 4.  Wrapper Terpadu: SHAPExplainer
# ─────────────────────────────────────────────
class SHAPExplainer:
    """
    Wrapper terpadu untuk semua model.
    Menerima objek HoaxDetectorInference dari predict.py.

    Contoh:
        from predict import HoaxDetectorInference
        from src.models.shap_explainer import SHAPExplainer

        detector = HoaxDetectorInference()
        xai = SHAPExplainer(detector)

        # Jelaskan satu teks
        result = xai.explain_all(teks, background_texts=train_texts[:50])
    """

    def __init__(self, detector):
        """
        Parameters
        ----------
        detector : HoaxDetectorInference
            Instance dari predict.py yang sudah load semua model.
        """
        self.detector = detector
        self._svm_xai = None
        self._cnn_xai = None
        self._bert_xai = None

    # ── SVM ──────────────────────────────────
    def _init_svm_xai(self):
        if self._svm_xai is None:
            if self.detector.svm is None:
                raise ValueError("Model SVM tidak tersedia di detector.")
            self._svm_xai = SVMSHAPExplainer(self.detector.svm)
        return self._svm_xai

    def explain_svm(self, text: str, top_n: int = 20, plot: bool = True,
                    save_path: str = None):
        """Jalankan SHAP untuk SVM+TF-IDF."""
        xai = self._init_svm_xai()
        result = xai.explain(text, top_n=top_n, plot=plot, save_path=save_path)
        self._print_result("SVM + TF-IDF", result["top_positive"], result["top_negative"])
        return result

    # ── TextCNN ──────────────────────────────
    def _init_cnn_xai(self, background_texts: list):
        if self._cnn_xai is None:
            if self.detector.cnn_model is None:
                raise ValueError("Model TextCNN tidak tersedia di detector.")
            self._cnn_xai = TextCNNSHAPExplainer(
                cnn_model=self.detector.cnn_model,
                cnn_vocab=self.detector.cnn_vocab,
                background_texts=background_texts,
                device=self.detector.device
            )
        return self._cnn_xai

    def explain_textcnn(self, text: str, background_texts: list,
                        top_n: int = 20, nsamples: int = 200,
                        plot: bool = True, save_path: str = None):
        """Jalankan SHAP untuk TextCNN."""
        xai = self._init_cnn_xai(background_texts)
        result = xai.explain(text, top_n=top_n, nsamples=nsamples,
                              plot=plot, save_path=save_path)
        self._print_result("TextCNN", result["top_positive"], result["top_negative"])
        return result

    # ── IndoBERT ─────────────────────────────
    def _init_bert_xai(self):
        if self._bert_xai is None:
            if self.detector.bert_model is None:
                raise ValueError("Model IndoBERT tidak tersedia di detector.")
            self._bert_xai = IndoBERTSHAPExplainer(
                bert_model=self.detector.bert_model,
                bert_tokenizer=self.detector.bert_tokenizer,
                device=self.detector.device
            )
        return self._bert_xai

    def explain_indobert(self, text: str, top_n: int = 20, max_evals: int = 500,
                          plot: bool = True, save_path: str = None):
        """Jalankan SHAP untuk IndoBERT."""
        xai = self._init_bert_xai()
        result = xai.explain(text, top_n=top_n, plot=plot,
                              save_path=save_path, max_evals=max_evals)
        self._print_result("IndoBERT", result["top_positive"], result["top_negative"])
        return result

    # ── Semua Model Sekaligus ─────────────────
    def explain_all(self, text: str, background_texts: list,
                    top_n: int = 15, output_dir: str = "results/shap"):
        """
        Jalankan SHAP untuk semua model yang tersedia.
        Simpan semua plot ke `output_dir`.

        Returns:
            dict: {'svm': ..., 'textcnn': ..., 'indobert': ...}
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        print("\n" + "="*65)
        print("         SHAP – EXPLAINABLE AI ANALYSIS")
        print("="*65)
        print(f"Teks: \"{text[:200]}{'...' if len(text) > 200 else ''}\"")
        print("="*65)

        # SVM
        if self.detector.svm is not None:
            print("\n[1/3] Menghitung SHAP untuk SVM + TF-IDF...")
            try:
                results["svm"] = self.explain_svm(
                    text, top_n=top_n, plot=True,
                    save_path=os.path.join(output_dir, "shap_svm.png")
                )
            except Exception as e:
                print(f"  [WARNING] SVM SHAP gagal: {e}")

        # TextCNN
        if self.detector.cnn_model is not None:
            print("\n[2/3] Menghitung SHAP untuk TextCNN (ini butuh waktu)...")
            try:
                results["textcnn"] = self.explain_textcnn(
                    text, background_texts=background_texts,
                    top_n=top_n, nsamples=150, plot=True,
                    save_path=os.path.join(output_dir, "shap_textcnn.png")
                )
            except Exception as e:
                print(f"  [WARNING] TextCNN SHAP gagal: {e}")

        # IndoBERT
        if self.detector.bert_model is not None:
            print("\n[3/3] Menghitung SHAP untuk IndoBERT...")
            try:
                results["indobert"] = self.explain_indobert(
                    text, top_n=top_n, max_evals=300, plot=True,
                    save_path=os.path.join(output_dir, "shap_indobert.png")
                )
            except Exception as e:
                print(f"  [WARNING] IndoBERT SHAP gagal: {e}")

        print("\n" + "="*65)
        print(f"Semua plot SHAP disimpan di folder: {output_dir}/")
        print("="*65 + "\n")
        return results

    # ── Utility ──────────────────────────────
    @staticmethod
    def _print_result(model_name: str, top_pos: list, top_neg: list):
        print(f"\n  ┌─ {model_name} ─────────────────────────────────")
        print("  │  🔴 Token pendorong HOAX (SHAP positif):")
        for t, v in top_pos[:10]:
            print(f"  │     {t:<25} +{v:.4f}")
        print("  │  🟢 Token pendorong NON-HOAX (SHAP negatif):")
        for t, v in top_neg[:10]:
            print(f"  │     {t:<25} {v:.4f}")
        print("  └" + "─"*50)
