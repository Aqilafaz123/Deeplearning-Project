import nbformat

notebook_path = 'perbandingan_3_arsitektur_indobert_cnn_svm.ipynb'
nb = nbformat.read(notebook_path, as_version=4)

# 1. Update Cell 3 (Markdown)
nb.cells[3].source = """## 2. Pemuatan Dataset & Exploratory Data Analysis (EDA)
Pada tahap ini, kita menggabungkan dataset archive politik dengan dataset multi-topik hasil live-scraping (3.995 data baru):
- **Dataset Archive Politik (`archive/Summarized/`)**:
  - Berita Valid (Label 0): CNN Indonesia, Kompas, Tempo
  - Berita Hoaks (Label 1): TurnBackHoax Mafindo
- **Dataset Baru Live-Scraped Multi-Topik (`dataset_multimedia/RAW/`)**:
  - Berita Valid (Label 0): Kompas (Kesehatan & Berita Nasional), Antara News (Kesehatan & Berita Nasional), CNN Indonesia
  - Berita Hoaks (Label 1): TurnBackHoax (Multi-kategori: Kesehatan, Penipuan/Bansos, Lowongan, Edukasi, Bencana), Liputan6 Cek Fakta (Terakreditasi IFCN)"""

# 2. Update Cell 4 (Code)
nb.cells[4].source = """# Konfigurasi sumber dataset: Archive Politik + Live Scraped (Multi-topik & Multi-media: 3.995 data baru)
data_configs = [
    # 1. Dataset Archive (Politik)
    {"file": "archive/Summarized/dataset_cnn_summarized.xlsx", "source": "CNN", "label": 0},
    {"file": "archive/Summarized/dataset_kompas_summarized.xlsx", "source": "Kompas", "label": 0},
    {"file": "archive/Summarized/dataset_tempo_summarized.xlsx", "source": "Tempo", "label": 0},
    {"file": "archive/Summarized/dataset_turnbackhoax_summarized.xlsx", "source": "TurnBackHoax", "label": 1},
    # 2. Dataset Baru Hasil Live-Scraping (Multi-topik & Multi-media)
    {"file": "dataset_multimedia/RAW/dataset_turnbackhoax_raw.xlsx", "source": "TurnBackHoax", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_liputan6_raw.xlsx", "source": "Liputan6", "label": 1},
    {"file": "dataset_multimedia/RAW/dataset_kompas_raw.xlsx", "source": "Kompas", "label": 0},
    {"file": "dataset_multimedia/RAW/dataset_antara_raw.xlsx", "source": "Antara", "label": 0},
    {"file": "dataset_multimedia/RAW/dataset_cnn_raw.xlsx", "source": "CNN", "label": 0},
]

def extract_best_text(row):
    \"\"\"Memilih teks terbaik yang tidak kosong dan bukan string 'nan'.\"\"\"
    candidates = ['summarized', 'FullText', 'cleaned', 'original', 'Narasi', 'Title']
    for col in candidates:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() != 'nan' and len(val) > 10:
                return val
    return ''

dfs = []
for cfg in data_configs:
    if not os.path.exists(cfg["file"]):
        print(f"Peringatan: File {cfg['file']} tidak ditemukan, dilewati.")
        continue
        
    df_temp = pd.read_excel(cfg["file"])
    df_temp['source'] = cfg["source"]
    df_temp['label'] = cfg["label"]
    df_temp['text'] = df_temp.apply(extract_best_text, axis=1)
    
    # Filter baris dengan teks valid (> 10 karakter)
    df_valid = df_temp[df_temp['text'].str.strip().str.len() > 10]
    dfs.append(df_valid[['text', 'label', 'source']])

df_all = pd.concat(dfs, ignore_index=True)

# Deduplikasi berbasis teks untuk menghindari artikel ganda
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"Total Sampel Data Bersih & Unik: {len(df_all):,}")
print(\"\\nDistribusi Label (0 = Non-Hoax, 1 = Hoax):\")
print(df_all['label'].value_counts())
print(\"\\nRincian per Sumber Media:\")
print(df_all['source'].value_counts())
df_all.head(3)"""

# 3. Update Cell 5 (Code)
nb.cells[5].source = """# Visualisasi Distribusi Kelas dan Sumber Data
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Grafik Distribusi Label
label_counts = df_all['label'].value_counts()
sns.barplot(x=['Non-Hoax (0)', 'Hoax (1)'], y=label_counts.values, ax=axes[0], palette=['#2b5c8f', '#d9534f'])
axes[0].set_title('Distribusi Kelas Berita', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Jumlah Data', fontsize=10)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 250, f'{v:,}', ha='center', fontweight='bold')

# Grafik Distribusi per Media
source_counts = df_all['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, ax=axes[1], palette='Blues_r')
axes[1].set_title('Distribusi Data per Sumber Media (Multi-Media)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Jumlah Data', fontsize=10)
axes[1].tick_params(axis='x', rotation=25)
for i, v in enumerate(source_counts.values):
    axes[1].text(i, v + 150, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()"""

# 4. Update Cell 7 (Code)
nb.cells[7].source = """from sklearn.model_selection import train_test_split

# Split 1: 80% Train, 20% (Val + Test)
train_df, val_test_df = train_test_split(
    df_all, 
    test_size=0.20, 
    random_state=SEED, 
    stratify=df_all['label']
)

# Split 2: 10% Val, 10% Test (50% dari sisa 20%)
val_df, test_df = train_test_split(
    val_test_df, 
    test_size=0.50, 
    random_state=SEED, 
    stratify=val_test_df['label']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Jumlah Data Train : {len(train_df):,} | Rasio Label: {dict(train_df['label'].value_counts())}")
print(f"Jumlah Data Val   : {len(val_df):,} | Rasio Label: {dict(val_df['label'].value_counts())}")
print(f"Jumlah Data Test  : {len(test_df):,} | Rasio Label: {dict(test_df['label'].value_counts())}")

# Simpan splits ke folder data_splits untuk reproduktifitas penelitian
os.makedirs('data_splits', exist_ok=True)
train_df.to_csv('data_splits/train_combined_35k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_combined_35k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_combined_35k.csv', index=False, encoding='utf-8')
print("Pemisahan data (train/val/test) berhasil disimpan di data_splits/!")"""

# Simpan kembali notebook
nbformat.write(nb, notebook_path)
print("Notebook 'perbandingan_3_arsitektur_indobert_cnn_svm.ipynb' berhasil diperbarui!")

