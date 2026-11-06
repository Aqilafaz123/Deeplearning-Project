import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SEED = 42

candidates = ['summarized', 'FullText', 'cleaned', 'original', 'Narasi', 'Title']

def extract_best_text(row):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() != 'nan' and len(val) > 10:
                return val
    return ''

archive_configs = [
    {'file': 'archive/Summarized/dataset_cnn_summarized.xlsx', 'source': 'CNN', 'label': 0},
    {'file': 'archive/Summarized/dataset_kompas_summarized.xlsx', 'source': 'Kompas', 'label': 0},
    {'file': 'archive/Summarized/dataset_tempo_summarized.xlsx', 'source': 'Tempo', 'label': 0},
    {'file': 'archive/Summarized/dataset_turnbackhoax_summarized.xlsx', 'source': 'TurnBackHoax', 'label': 1},
]

dfs = []
for cfg in archive_configs:
    if not os.path.exists(cfg['file']):
        continue
    df_temp = pd.read_excel(cfg['file'])
    df_temp['source'] = cfg['source']
    df_temp['label'] = cfg['label']
    df_temp['text'] = df_temp.apply(extract_best_text, axis=1)
    df_valid = df_temp[df_temp['text'].str.strip().str.len() > 10]
    dfs.append(df_valid[['text', 'label', 'source']])

# Load freshly scraped multi-media dataset (10,118 rows)
csv_file = 'dataset_multimedia/RAW/dataset_multimedia_combined.csv'
if os.path.exists(csv_file):
    df_csv = pd.read_csv(csv_file)
    df_csv['text'] = df_csv.apply(extract_best_text, axis=1)
    df_csv['label'] = df_csv['hoax'].astype(int)
    df_csv['source'] = df_csv['Source']
    df_valid_csv = df_csv[df_csv['text'].str.strip().str.len() > 10]
    dfs.append(df_valid_csv[['text', 'label', 'source']])

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"Total Sampel Data Bersih & Unik: {len(df_all):,}")
print("\nDistribusi Label (0 = Non-Hoax, 1 = Hoax):")
print(df_all['label'].value_counts())
print("\nRincian per Sumber Media:")
print(df_all['source'].value_counts())

train_df, val_test_df = train_test_split(df_all, test_size=0.20, random_state=SEED, stratify=df_all['label'])
val_df, test_df = train_test_split(val_test_df, test_size=0.50, random_state=SEED, stratify=val_test_df['label'])

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\nTrain: {len(train_df):,} | {dict(train_df['label'].value_counts())}")
print(f"Val:   {len(val_df):,} | {dict(val_df['label'].value_counts())}")
print(f"Test:  {len(test_df):,} | {dict(test_df['label'].value_counts())}")

os.makedirs('data_splits', exist_ok=True)
train_df.to_csv('data_splits/train_combined_41k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_combined_41k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_combined_41k.csv', index=False, encoding='utf-8')

# Also update train_combined_35k.csv or link them
train_df.to_csv('data_splits/train_combined_35k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_combined_35k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_combined_35k.csv', index=False, encoding='utf-8')

# Save master dataset 41k
df_all.to_csv('data_splits/dataset_master_combined_41k.csv', index=False, encoding='utf-8')
print("Pemisahan data berhasil disimpan di data_splits/!")

