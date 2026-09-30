import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from src.text_preprocessing import clean_text_id


class SVMTextClassifier:
    """
    TF-IDF + Calibrated Linear Support Vector Machine for text classification.
    CalibratedClassifierCV enables probability estimates (predict_proba) for ROC-AUC.
    """
    def __init__(self, max_features=10000, ngram_range=(1, 2), C=1.0, random_state=42):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state
        
        base_svc = LinearSVC(C=self.C, random_state=self.random_state, max_iter=2000)
        self.model = CalibratedClassifierCV(estimator=base_svc, cv=3)
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text_id,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True
        )
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.model)
        ])

    def fit(self, train_texts, train_labels):
        print(f"[SVM] Fitting TF-IDF Vectorizer (max_features={self.max_features}, ngrams={self.ngram_range}) and SVM...")
        self.pipeline.fit(train_texts, train_labels)
        print("[SVM] Training completed.")

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"[SVM] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance
