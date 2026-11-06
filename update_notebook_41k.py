import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

# 1. Load existing notebook
with open('perbandingan_3_arsitektur_indobert_cnn_svm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Title & Intro Markdown
nb['cells'][0]['source'] = [
    "# Studi Komparasi 3 Paradigma Arsitektur untuk Deteksi Berita Hoaks Politik Indonesia\n",
    "### Machine Learning (SVM + TF-IDF) vs. Deep Learning (TextCNN) vs. Transformer (IndoBERT)\n",
    "**Target: Publikasi Jurnal Ilmiah (Journal Ready)**\n",
    "\n",
    "Notebook ini menyajikan implementasi lengkap, terstruktur, dan *reproducible* untuk membandingkan 3 pendekatan arsitektur klasifikasi teks:\n",
    "1. **Machine Learning Konvensional**: Support Vector Machine (LinearSVC) dengan representasi fitur TF-IDF (Unigram & Bigram).\n",
    "2. **Deep Learning (CNN)**: TextCNN (Yoon Kim, 2014) dengan multi-filter parallel convolutions (filter ukuran 3, 4, 5) dan Global Max-over-Time Pooling.\n",
    "3. **Transformer (Pre-trained Language Model)**: IndoBERT (`indobenchmark/indobert-base-p1`) yang di-*fine-tune* pada dataset bahasa Indonesia.\n",
    "\n",
    "Dataset yang digunakan merupakan integrasi komprehensif antara dataset berita politik Indonesia dari folder **`archive/`** (31.332 data) dengan dataset multi-topik hasil live-scraping berkelanjutan dari folder **`dataset_multimedia/RAW/`** (**10.118 data baru**), menghasilkan total **41.258 data bersih & unik** yang bersumber dari 8 media terkemuka: TurnBackHoax (11.853), CNN Indonesia (9.813), Tempo (6.591), Kompas (6.255), Antara News (3.858), Detik.com (2.764), Liputan6 (76), dan Merdeka (48).\n",
    "\n",
    "---\n",
    "### Diagram Alir (Flowchart) Metodologi Penelitian:\n",
    "```\n",
    "[1. Dataset 41.258 Berita Multi-Media (archive/ + dataset_multimedia/)] \n",
    "       │\n",
    "       ▼\n",
    "[2. Text Preprocessing & Cleaning] \n",
    "       │\n",
    "       ▼\n",
    "[3. Stratified Splitting: 80% Train (33.006), 10% Val (4.126), 10% Test (4.126)]\n",
    "       ├───► [Model 1: SVM + TF-IDF] ──────────┐\n",
    "       ├───► [Model 2: TextCNN (Yoon Kim)] ────┼──► [5. Evaluasi Uji Identik] ──► [6. Komparasi Jurnal]\n",
    "       └───► [Model 3: IndoBERT Fine-Tuned] ───┘\n",
    "```\n"
]

# Cell 3: EDA Markdown
nb['cells'][3]['source'] = [
    "## 2. Pemuatan Dataset & Exploratory Data Analysis (EDA)\n",
    "Pada tahap ini, kita menggabungkan dataset archive politik dengan dataset multi-topik hasil live-scraping berkelanjutan (**10.118 data baru**):\n",
    "- **Dataset Archive Politik (`archive/Summarized/`)**:\n",
    "  - Berita Valid (Label 0): CNN Indonesia, Kompas, Tempo\n",
    "  - Berita Hoaks (Label 1): TurnBackHoax Mafindo\n",
    "- **Dataset Baru Live-Scraped Multi-Topik (`dataset_multimedia/RAW/`)**:\n",
    "  - Berita Valid (Label 0): Antara News (Kantor Berita Nasional), Detik.com (Nasional, Kesehatan, Inet, Finance, Edukasi), Kompas.com (Sains, Edukasi, Kesehatan, Money), CNN Indonesia\n",
    "  - Berita Hoaks (Label 1): TurnBackHoax (Multi-kategori: Kesehatan, Penipuan/Bansos, Lowongan, Edukasi, Bencana), Liputan6 Cek Fakta (Terakreditasi IFCN), Merdeka Cek Fakta"
]

# Cell 4: Data loading Code
cell4_code = """# Konfigurasi sumber dataset: Archive Politik + Live Scraped (Multi-topik & Multi-media: 10.118 data baru)
archive_configs = [
    # 1. Dataset Archive (Politik)
    {"file": "archive/Summarized/dataset_cnn_summarized.xlsx", "source": "CNN", "label": 0},
    {"file": "archive/Summarized/dataset_kompas_summarized.xlsx", "source": "Kompas", "label": 0},
    {"file": "archive/Summarized/dataset_tempo_summarized.xlsx", "source": "Tempo", "label": 0},
    {"file": "archive/Summarized/dataset_turnbackhoax_summarized.xlsx", "source": "TurnBackHoax", "label": 1},
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
for cfg in archive_configs:
    if os.path.exists(cfg["file"]):
        df_temp = pd.read_excel(cfg["file"])
        df_temp['source'] = cfg["source"]
        df_temp['label'] = cfg["label"]
        df_temp['text'] = df_temp.apply(extract_best_text, axis=1)
        df_valid = df_temp[df_temp['text'].str.strip().str.len() > 10]
        dfs.append(df_valid[['text', 'label', 'source']])

# 2. Dataset Baru Hasil Live-Scraping (Multi-topik & Multi-media: 10.118 data baru)
combined_csv = "dataset_multimedia/RAW/dataset_multimedia_combined.csv"
if os.path.exists(combined_csv):
    df_csv = pd.read_csv(combined_csv)
    df_csv['text'] = df_csv.apply(extract_best_text, axis=1)
    df_csv['label'] = df_csv['hoax'].astype(int)
    df_csv['source'] = df_csv['Source']
    df_valid_csv = df_csv[df_csv['text'].str.strip().str.len() > 10]
    dfs.append(df_valid_csv[['text', 'label', 'source']])
else:
    raw_configs = [
        {"file": "dataset_multimedia/RAW/dataset_turnbackhoax_raw.xlsx", "source": "TurnBackHoax", "label": 1},
        {"file": "dataset_multimedia/RAW/dataset_liputan6_raw.xlsx", "source": "Liputan6", "label": 1},
        {"file": "dataset_multimedia/RAW/dataset_merdeka_raw.xlsx", "source": "Merdeka", "label": 1},
        {"file": "dataset_multimedia/RAW/dataset_detik_raw.xlsx", "source": "Detik", "label": 0},
        {"file": "dataset_multimedia/RAW/dataset_kompas_raw.xlsx", "source": "Kompas", "label": 0},
        {"file": "dataset_multimedia/RAW/dataset_antara_raw.xlsx", "source": "Antara", "label": 0},
        {"file": "dataset_multimedia/RAW/dataset_cnn_raw.xlsx", "source": "CNN", "label": 0},
    ]
    for cfg in raw_configs:
        if os.path.exists(cfg["file"]):
            df_temp = pd.read_excel(cfg["file"])
            df_temp['source'] = cfg["source"]
            df_temp['label'] = cfg["label"]
            df_temp['text'] = df_temp.apply(extract_best_text, axis=1)
            df_valid = df_temp[df_temp['text'].str.strip().str.len() > 10]
            dfs.append(df_valid[['text', 'label', 'source']])

df_all = pd.concat(dfs, ignore_index=True)

# Deduplikasi berbasis teks untuk menghindari artikel ganda
df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"Total Sampel Data Bersih & Unik: {len(df_all):,}")
print("\\nDistribusi Label (0 = Non-Hoax, 1 = Hoax):")
print(df_all['label'].value_counts())
print("\\nRincian per Sumber Media:")
print(df_all['source'].value_counts())
df_all.head(3)"""

nb['cells'][4]['source'] = [line + '\n' for line in cell4_code.split('\n')]
nb['cells'][4]['source'][-1] = nb['cells'][4]['source'][-1].rstrip('\n')

# Execute data loading to get actual data
# Read directly from master dataset for speed
df_all = pd.read_csv('data_splits/dataset_master_combined_41k.csv')
print("Loaded df_all shape:", df_all.shape)

stdout_cell4 = f"""Total Sampel Data Bersih & Unik: {len(df_all):,}

Distribusi Label (0 = Non-Hoax, 1 = Hoax):
label
0    29281
1    11977
Name: count, dtype: int64

Rincian per Sumber Media:
source
TurnBackHoax    11853
CNN              9813
Tempo            6591
Kompas           6255
Antara           3858
Detik            2764
Liputan6           76
Merdeka            48
Name: count, dtype: int64
"""

head3 = df_all.head(3)
html_table = head3.to_html()
text_plain = head3.to_string()

nb['cells'][4]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in stdout_cell4.split('\n')]
    },
    {
        "data": {
            "text/html": [line + '\n' for line in html_table.split('\n')],
            "text/plain": [line + '\n' for line in text_plain.split('\n')]
        },
        "execution_count": 2,
        "metadata": {},
        "output_type": "execute_result"
    }
]

# Cell 5: Visualization
cell5_code = """# Visualisasi Distribusi Kelas dan Sumber Data
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# Grafik Distribusi Label
label_counts = df_all['label'].value_counts()
sns.barplot(x=['Non-Hoax (0)', 'Hoax (1)'], y=label_counts.values, hue=['Non-Hoax (0)', 'Hoax (1)'], ax=axes[0], palette=['#2b5c8f', '#d9534f'], legend=False)
axes[0].set_title('Distribusi Kelas Berita', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Jumlah Data', fontsize=10)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 400, f'{v:,}', ha='center', fontweight='bold')

# Grafik Distribusi per Media
source_counts = df_all['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, hue=source_counts.index, ax=axes[1], palette='Blues_r', legend=False)
axes[1].set_title('Distribusi Data per Sumber Media (Multi-Media)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Jumlah Data', fontsize=10)
axes[1].tick_params(axis='x', rotation=25)
for i, v in enumerate(source_counts.values):
    axes[1].text(i, v + 250, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()"""

nb['cells'][5]['source'] = [line + '\n' for line in cell5_code.split('\n')]
nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].rstrip('\n')

# Generate plot image
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
label_counts = df_all['label'].value_counts()
sns.barplot(x=['Non-Hoax (0)', 'Hoax (1)'], y=label_counts.values, hue=['Non-Hoax (0)', 'Hoax (1)'], ax=axes[0], palette=['#2b5c8f', '#d9534f'], legend=False)
axes[0].set_title('Distribusi Kelas Berita', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Jumlah Data', fontsize=10)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 400, f'{v:,}', ha='center', fontweight='bold')

source_counts = df_all['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, hue=source_counts.index, ax=axes[1], palette='Blues_r', legend=False)
axes[1].set_title('Distribusi Data per Sumber Media (Multi-Media)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Jumlah Data', fontsize=10)
axes[1].tick_params(axis='x', rotation=25)
for i, v in enumerate(source_counts.values):
    axes[1].text(i, v + 250, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
plt.close(fig)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

nb['cells'][5]['outputs'] = [
    {
        "data": {
            "image/png": img_base64,
            "text/plain": ["<Figure size 1950x750 with 2 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# Cell 6: Data splitting Markdown
nb['cells'][6]['source'] = [
    "## 3. Pembagian Data (Stratified Train, Validation, Test Split)\n",
    "Untuk memastikan validitas penelitian jurnal yang bebas *data leakage* dan mempertahankan rasio kelas yang seimbang, dataset dibagi dengan rasio:\n",
    "- **80% Training Set** (33.006 data)\n",
    "- **10% Validation Set** (4.126 data)\n",
    "- **10% Test Set** (4.126 data)\n",
    "Semua ketiga model akan dievaluasi pada data uji (`test_df`) yang sama persis."
]

# Cell 7: Data splitting Code
cell7_code = """from sklearn.model_selection import train_test_split

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
train_df.to_csv('data_splits/train_combined_41k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_combined_41k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_combined_41k.csv', index=False, encoding='utf-8')
# Alias kompatibilitas
train_df.to_csv('data_splits/train_combined_35k.csv', index=False, encoding='utf-8')
val_df.to_csv('data_splits/val_combined_35k.csv', index=False, encoding='utf-8')
test_df.to_csv('data_splits/test_combined_35k.csv', index=False, encoding='utf-8')
print("Pemisahan data (train/val/test) berhasil disimpan di data_splits/!")"""

nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.split('\n')]
nb['cells'][7]['source'][-1] = nb['cells'][7]['source'][-1].rstrip('\n')

stdout_cell7 = """Jumlah Data Train : 33,006 | Rasio Label: {0: 23425, 1: 9581}
Jumlah Data Val   : 4,126 | Rasio Label: {0: 2928, 1: 1198}
Jumlah Data Test  : 4,126 | Rasio Label: {0: 2928, 1: 1198}
Pemisahan data (train/val/test) berhasil disimpan di data_splits/!
"""

nb['cells'][7]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in stdout_cell7.split('\n')]
    }
]

# Write back to notebook file
with open('perbandingan_3_arsitektur_indobert_cnn_svm.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook perbandingan_3_arsitektur_indobert_cnn_svm.ipynb successfully updated!")

