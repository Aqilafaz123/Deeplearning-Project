import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SEED = 42

csv_path = 'dataset_multimedia/RAW/dataset_multimedia_combined.csv'
df = pd.read_csv(csv_path)

candidates = ['FullText', 'Narasi', 'Title']
def extract_text(r):
    for c in candidates:
        if c in r and pd.notna(r[c]):
            v = str(r[c]).strip()
            if v and v.lower() != 'nan' and len(v) > 10:
                return v
    return ''

df['text'] = df.apply(extract_text, axis=1)
df['label'] = df['hoax'].astype(int)
df['source'] = df['Source']

df_all = df[df['text'].str.strip().str.len() > 10].reset_index(drop=True)
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)
df_all = df_all[['text', 'label', 'source']]

print("=== DATASET HASIL SCRAPING MURNI (TANPA ARCHIVE) ===")
print(f"Total Sampel Data Bersih & Unik : {len(df_all):,}")
print("\nDistribusi Label (0 = Non-Hoax, 1 = Hoax):")
print(df_all['label'].value_counts())
print("\nRincian per Sumber Media:")
print(df_all['source'].value_counts())

train_df, val_test_df = train_test_split(df_all, test_size=0.20, random_state=SEED, stratify=df_all['label'])
val_df, test_df = train_test_split(val_test_df, test_size=0.50, random_state=SEED, stratify=val_test_df['label'])

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\nTrain (80%): {len(train_df):,} | {dict(train_df['label'].value_counts())}")
print(f"Val   (10%): {len(val_df):,} | {dict(val_df['label'].value_counts())}")
print(f"Test  (10%): {len(test_df):,} | {dict(test_df['label'].value_counts())}")

os.makedirs('data_splits', exist_ok=True)
train_df.to_csv('data_splits/train_scraped_10k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_scraped_10k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_scraped_10k.csv', index=False, encoding='utf-8')
df_all.to_csv('data_splits/dataset_master_scraped_10k.csv', index=False, encoding='utf-8')
print("File partisi berhasil disimpan di data_splits/ (train/val/test_scraped_10k.csv)!")

