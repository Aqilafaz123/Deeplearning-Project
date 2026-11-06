import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re

test_df = pd.read_csv('data_splits/test_scraped_10k.csv')
train_df = pd.read_csv('data_splits/train_scraped_10k.csv')

print(f"Jumlah Test: {len(test_df)} (Fakta: {(test_df['label']==0).sum()}, Hoaks: {(test_df['label']==1).sum()})")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

vec = TfidfVectorizer(preprocessor=clean_text, max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vec.fit_transform(train_df['text'])
X_test = vec.transform(test_df['text'])

svm = LinearSVC(C=1.0, random_state=42)
svm.fit(X_train, train_df['label'])
preds = svm.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_df['label'], preds)
print("\nConfusion Matrix SVM:")
print(f"  True Neg (Fakta benar): {cm[0,0]} / {(test_df['label']==0).sum()}")
print(f"  False Pos (Fakta dikira Hoaks): {cm[0,1]}")
print(f"  False Neg (Hoaks lolos / dikira Fakta): {cm[1,0]}")
print(f"  True Pos (Hoaks terdeteksi): {cm[1,1]} / {(test_df['label']==1).sum()}")

# Inspect top features
coefs = svm.coef_[0]
feature_names = np.array(vec.get_feature_names_out())
top_hoax_idx = np.argsort(coefs)[-20:][::-1]
top_fact_idx = np.argsort(coefs)[:20]

print("\n=== TOP 20 KATA PENENTU HOAKS (Positif) ===")
for idx in top_hoax_idx:
    print(f"  {feature_names[idx]:<25}: {coefs[idx]:.4f}")

print("\n=== TOP 20 KATA PENENTU FAKTA (Negatif) ===")
for idx in top_fact_idx:
    print(f"  {feature_names[idx]:<25}: {coefs[idx]:.4f}")

