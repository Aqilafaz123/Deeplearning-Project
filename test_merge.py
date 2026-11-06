import pandas as pd

sources = [
    # Archive sources
    {'file': 'archive/Summarized/dataset_cnn_summarized.xlsx', 'source': 'CNN', 'label': 0},
    {'file': 'archive/Summarized/dataset_kompas_summarized.xlsx', 'source': 'Kompas', 'label': 0},
    {'file': 'archive/Summarized/dataset_tempo_summarized.xlsx', 'source': 'Tempo', 'label': 0},
    {'file': 'archive/Summarized/dataset_turnbackhoax_summarized.xlsx', 'source': 'TurnBackHoax', 'label': 1},
    # 3,995 Live-scraped sources
    {'file': 'dataset_multimedia/RAW/dataset_turnbackhoax_raw.xlsx', 'source': 'TurnBackHoax', 'label': 1},
    {'file': 'dataset_multimedia/RAW/dataset_liputan6_raw.xlsx', 'source': 'Liputan6', 'label': 1},
    {'file': 'dataset_multimedia/RAW/dataset_kompas_raw.xlsx', 'source': 'Kompas', 'label': 0},
    {'file': 'dataset_multimedia/RAW/dataset_antara_raw.xlsx', 'source': 'Antara', 'label': 0},
    {'file': 'dataset_multimedia/RAW/dataset_cnn_raw.xlsx', 'source': 'CNN', 'label': 0},
]

def extract_text(df):
    candidates = ['summarized', 'FullText', 'cleaned', 'original', 'Narasi', 'Title']
    def row_picker(row):
        for col in candidates:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val.lower() != 'nan' and len(val) > 10:
                    return val
        return ''
    return df.apply(row_picker, axis=1)

dfs = []
for src in sources:
    df_temp = pd.read_excel(src['file'])
    df_temp['source'] = src['source']
    df_temp['label'] = src['label']
    df_temp['text'] = extract_text(df_temp)
    valid_count = (df_temp['text'].str.strip().str.len() > 10).sum()
    print(f"{src['source']:15} ({src['file']}): {len(df_temp)} rows -> {valid_count} valid")
    dfs.append(df_temp[['text', 'label', 'source']])

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[df_all['text'].str.strip().str.len() > 10].reset_index(drop=True)
print('\nTotal sebelum deduplikasi:', len(df_all))
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)
print('Total setelah deduplikasi:', len(df_all))
print('\nDistribusi Label:')
print(df_all['label'].value_counts())
print('\nDistribusi Media:')
print(df_all['source'].value_counts())

