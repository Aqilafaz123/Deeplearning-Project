import os
import json


class GenerativeFactChecker:
    """
    Generative AI Reasoner (Explainable AI / XAI) for political fake news analysis.
    Supports Google Gemini API with fallback to Groq / OpenAI-compatible APIs.
    """
    def __init__(self, api_key=None, provider="gemini"):
        self.provider = provider.lower()
        # API key diambil dari environment variable, jangan hardcode di sini!
        # Set: export GEMINI_API_KEY="your_api_key_here"
        if not api_key or "MASUKKAN_" in str(api_key):
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.api_key = api_key

    def explain(self, news_text, predicted_label="HOAX", confidence=0.99):
        """
        Generates comprehensive fact-checking reasoning for a given news text
        combined with the prediction result from discriminative models (IndoBERT/SVM).
        """
        if not self.api_key:
            return (
                "[PERINGATAN] API Key belum diatur.\n"
                "Untuk mengaktifkan penalaran Generative LLM, masukkan API Key Anda:\n"
                "  - Cara 1: detector.print_prediction(teks, explain=True, api_key='AIzaSy...')\n"
                "  - Cara 2: export GEMINI_API_KEY='AIzaSy...'\n"
                "API Key Google Gemini bisa didapatkan gratis di: https://aistudio.google.com/app/apikey"
            )

        prompt = f"""Anda adalah seorang analis jurnalisme investigatif dan pemeriksa fakta (fact-checker) profesional politik Indonesia.

Tugas Anda adalah memberikan analisis penalaran berbasis fakta (Explainable AI Reasoning) terhadap teks berita berikut.

HASIL DETEKSI DEEP LEARNING (IndoBERT):
- Prediksi Model: {predicted_label}
- Tingkat Keyakinan (Confidence): {confidence*100:.2f}%

TEKS BERITA YANG DIANALISIS:
\"\"\"{news_text}\"\"\"

Berikan hasil analisis yang kritis, objektif, dan terstruktur dengan format berikut:
1. **Ringkasan Inti Klaim**: (1-2 kalimat mengenai apa yang diklaim berita)
2. **Kesesuaian dengan Fakta Resmi**: (Apakah klaim ini sesuai dengan regulasi/pernyataan kementerian/lembaga resmi terkait?)
3. **Analisis Indikator Bahasa & Pola Berita**: (Jelaskan ciri linguistik, misalnya gaya bahasa sensasional, minimnya atribusi narasumber, atau kesesuaian gaya berita jurnalistik formal)
4. **Kesimpulan Akhir & Saran Verifikasi**: (Rekomendasi tindakan bagi masyarakat/pembaca untuk memverifikasi kebenaran berita ini)
"""

        try:
            if self.provider == "gemini":
                from google import genai
                client = genai.Client(api_key=self.api_key)
                models_to_try = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
                last_err = None
                for m in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=m,
                            contents=prompt
                        )
                        return response.text
                    except Exception as err:
                        last_err = err
                        continue
                raise last_err or Exception("Semua model gagal dihubungi.")
            else:
                return "Provider belum didukung."
        except Exception as e:
            return f"[ERROR] Terjadi kesalahan saat memanggil Generative LLM: {str(e)}"
