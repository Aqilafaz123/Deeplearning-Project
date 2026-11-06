import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

data_configs = [
    # --- Dataset Archive (Politik) ---
    {"file": "archive/Summarized/dataset_cnn_summarized.xlsx", "source": "CNN", "label": 0},
    {"file": "archive/Summarized/dataset_kompas_summarized.xlsx", "source": "Kompas", "label": 0},
    {"file": "archive/Summarized/dataset_tempo_summarized.xlsx", "source": "Tempo", "label": 0},
    {"file": "archive/Summarized/dataset_turnbackhoax_summarized.xlsx", "source": "TurnBackHoax", "label": 1},
    # --- Dataset Baru Hasil Live-Scraping (Multi-topik & Multi-media: 3.995 data) ---
    {"file": "dataset_multimedia/RAW/dataset_turnbackhoax_raw.xlsx", "source": "TurnBackHoax", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_liputan6_raw.xlsx", "source": "Liputan6", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_kompas_raw.xlsx", "source": "Kompas", "label": 0},
    {"file": "dataset_multimedia/RAW/dataset_antara_raw.xlsx", "source": "Antara", "label": 0},
    {"file": "dataset_multimedia/RAW/dataset_cnn_raw.xlsx", "source": "CNN", "label": 0},
]

def extract_best_text(row):
    candidates = ['summarized', 'FullText', 'cleaned', 'original', 'Narasi', 'Title']
    for col in candidates:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() != 'nan' and len(val) > 10:
                return val
    return ''

print("Memuat dan menggabungkan semua dataset...")
dfs = []
for cfg in data_configs:
    if not os.path.exists(cfg["file"]):
        print(f"Peringatan: File {cfg['file']} tidak ditemukan, dilewati.")
        continue
    
    df_temp = pd.read_excel(cfg["file"])
    df_temp['source'] = cfg["source"]
    df_temp['label'] = cfg["label"]
    df_temp['text'] = df_temp.apply(extract_best_text, axis=1)
    
    df_valid = df_temp[df_temp['text'].str.strip().str.len() > 10]
    print(f"  + {cfg['source']:15} | File: {cfg['file']:55} | Baris valid: {len(df_valid)}")
    dfs.append(df_valid[['text', 'label', 'source']])

df_all = pd.concat(dfs, ignore_index=True)
total_before_dedup = len(df_all)
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"\nTotal Sebelum Deduplikasi : {total_before_dedup}")
print(f"Total Setelah Deduplikasi : {len(df_all)}")
print("\nDistribusi Label (0 = Non-Hoax, 1 = Hoax):")
print(df_all['label'].value_counts())
print("\nRincian per Sumber Media:")
print(df_all['source'].value_counts())

# Stratified Split 80:10:10
train_df, val_test_df = train_test_split(
    df_all, 
    test_size=0.20, 
    random_state=SEED, 
    stratify=df_all['label']
)

val_df, test_df = train_test_split(
    val_test_df, 
    test_size=0.50, 
    random_state=SEED, 
    stratify=val_test_df['label']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\nJumlah Data Train : {len(train_df)} | Rasio Label: {dict(train_df['label'].value_counts())}")
print(f"Jumlah Data Val   : {len(val_df)} | Rasio Label: {dict(val_df['label'].value_counts())}")
print(f"Jumlah Data Test  : {len(test_df)} | Rasio Label: {dict(test_df['label'].value_counts())}")

# Simpan ke data_splits
os.makedirs("data_splits", exist_ok=True)
train_df.to_csv("data_splits/train_combined_35k.csv", index=False, encoding='utf-8')
val_df.to_csv("data_splits/val_combined_35k.csv", index=False, encoding='utf-8')
test_df.to_csv("data_splits/test_combined_35k.csv", index=False, encoding='utf-8')
print("\nData splits berhasil disimpan ke folder data_splits/ (train/val/test_combined_35k.csv)!")

