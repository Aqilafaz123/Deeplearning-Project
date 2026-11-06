"""
Script Generator Dataset Multi-Topik 10.000 Data (Balanced 50:50)
Topik mencakup:
  1. Kesehatan & Medis (Obat, Penyakit, Vaksin, Ginjal, Diabetes, dll.)
  2. Bencana Alam & Cuaca (Gempa, Banjir, Erupsi, Tsunami, BMKG)
  3. Penipuan & Finansial (Bansos palsu, Undian/Giveaway, Phising, Lowongan)
  4. Sosial, Budaya & Edukasi (Mitos, Peristiwa Viral, Sains, Tokoh)
  5. Politik & Pemerintahan (Proporsional terdistribusi)

Struktur Output:
  - 5.000 Hoaks (Label: 1)
  - 5.000 Fakta / Asli (Label: 0)
  - Train: 7.000 (70%) | Val: 1.500 (15%) | Test: 1.500 (15%)
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.text_preprocessing import clean_text_id

# Pastikan unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def categorize_text(title, text=""):
    """
    Mengelompokkan artikel ke dalam kategori topik berdasarkan kata kunci.
    """
    content = f"{str(title)} {str(text)}".lower()
    
    # 1. Kesehatan
    if any(k in content for k in [
        'kesehatan', 'obat', 'penyakit', 'vaksin', 'covid', 'corona', 'dokter', 
        'kanker', 'diabetes', 'darah', 'rumah sakit', 'virus', 'stroke', 'gula', 
        'imunisasi', 'gizi', 'diet', 'medis', 'bpjs', 'vitamin', 'gejala'
    ]):
        return "Kesehatan"
        
    # 2. Bencana Alam
    if any(k in content for k in [
        'gempa', 'tsunami', 'gunung', 'banjir', 'bencana', 'erupsi', 'bmkg', 
        'longsor', 'cuaca ekstrem', 'puting beliung', 'karhutla'
    ]):
        return "Bencana Alam"
        
    # 3. Penipuan & Finansial
    if any(k in content for k in [
        'penipuan', 'bantuan', 'dana', 'hadiah', 'undian', 'giveaway', 'pulsa', 
        'link', 'tautan', 'blt', 'bansos', 'lowongan', 'rekrutmen', 'pinjol', 
        'investasi bodong', 'saldo'
    ]):
        return "Penipuan & Bansos"
        
    # 4. Politik & Pemerintahan
    if any(k in content for k in [
        'presiden', 'prabowo', 'jokowi', 'ganjar', 'anies', 'gibran', 'dpr', 
        'kpu', 'pemilu', 'pilpres', 'pilkada', 'partai', 'pdi', 'gerindra', 
        'golkar', 'nasdem', 'menteri', 'bawaslu', 'uu '
    ]):
        return "Politik"
        
    # 5. Sosial & Peristiwa Umum
    return "Sosial & Umum"


def extract_hoax_data(target_count=5000):
    """
    Mengambil dan membersihkan 5.000 data hoaks dari arsip TurnBackHoax dan Komdigi.
    """
    print("\n[1/3] Mengumpulkan 5.000 data hoaks (Label: 1)...", flush=True)
    hoax_list = []
    
    # Sumber A: TurnBackHoax Archive Summarized
    tbh_path = r"archive\Summarized\dataset_turnbackhoax_summarized.xlsx"
    if os.path.exists(tbh_path):
        print(f"  -> Memuat data dari: {tbh_path} ...", flush=True)
        df_tbh = pd.read_excel(tbh_path)
        
        # Prioritaskan data non-politik terlebih dahulu (politik == 0)
        # lalu sisipkan porsi politik yang proporsional
        df_non_pol = df_tbh[df_tbh['politik'] == 0].copy()
        df_pol = df_tbh[df_tbh['politik'] == 1].copy()
        
        print(f"     Tersedia di TBH: {len(df_non_pol)} non-politik & {len(df_pol)} politik", flush=True)
        
        # Ambil ~4.000 non-politik dan ~800 politik
        sample_non_pol = df_non_pol.sample(n=min(len(df_non_pol), 4000), random_state=42)
        sample_pol = df_pol.sample(n=min(len(df_pol), 800), random_state=42)
        
        combined_tbh = pd.concat([sample_non_pol, sample_pol], ignore_index=True)
        
        for _, row in combined_tbh.iterrows():
            title = str(row.get('title', ''))
            # Gunakan narasi atau original text
            raw_text = str(row.get('original', ''))
            if len(raw_text) < 50:
                raw_text = str(row.get('cleaned', title))
                
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                cat = categorize_text(title, raw_text)
                hoax_list.append({
                    'title': title,
                    'text': cleaned,
                    'label': 1,
                    'category': cat,
                    'source': 'TurnBackHoax'
                })

    # Sumber B: Komdigi Hoaks CSV
    komdigi_path = "komdigi_hoaks.csv"
    if os.path.exists(komdigi_path):
        print(f"  -> Memuat data tambahan dari: {komdigi_path} ...", flush=True)
        df_komdigi = pd.read_csv(komdigi_path)
        for _, row in df_komdigi.iterrows():
            title = str(row.get('title', ''))
            raw_text = str(row.get('body_text', ''))
            if len(raw_text) < 50:
                raw_text = str(row.get('excerpt', title))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                cat = categorize_text(title, raw_text)
                hoax_list.append({
                    'title': title,
                    'text': cleaned,
                    'label': 1,
                    'category': cat,
                    'source': 'Kominfo/Komdigi'
                })
                
    # Sumber C: Data scraper kesehatan baru jika ada
    scraped_tbh = r"dataset_kesehatan\RAW\dataset_turnbackhoax_raw.xlsx"
    if os.path.exists(scraped_tbh):
        df_sc = pd.read_excel(scraped_tbh)
        for _, row in df_sc.iterrows():
            title = str(row.get('Title', ''))
            raw_text = str(row.get('FullText', ''))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                hoax_list.append({
                    'title': title,
                    'text': cleaned,
                    'label': 1,
                    'category': 'Kesehatan',
                    'source': 'TurnBackHoax-Live'
                })

    df_hoax = pd.DataFrame(hoax_list)
    # Deduplikasi
    df_hoax = df_hoax.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"  -> Total kandidat hoaks unik terkumpul: {len(df_hoax)} baris", flush=True)
    
    # Ambil tepat target_count
    if len(df_hoax) > target_count:
        df_hoax = df_hoax.sample(n=target_count, random_state=42).reset_index(drop=True)
        
    print(f"  [OK] Berhasil mengamankan tepat {len(df_hoax)} data hoaks (Label: 1)\n", flush=True)
    return df_hoax


def extract_valid_data(target_count=5000):
    """
    Mengambil dan membersihkan 5.000 data berita asli dari CNN, Tempo, Kompas.
    """
    print("[2/3] Mengumpulkan 5.000 data berita asli / fakta (Label: 0)...", flush=True)
    valid_list = []
    
    sources = [
        ("CNN Indonesia", r"archive\Summarized\dataset_cnn_summarized.xlsx", 2000),
        ("Tempo", r"archive\Summarized\dataset_tempo_summarized.xlsx", 1800),
        ("Kompas", r"archive\Summarized\dataset_kompas_summarized.xlsx", 1500)
    ]
    
    for source_name, file_path, n_samples in sources:
        if os.path.exists(file_path):
            print(f"  -> Memuat data dari: {file_path} ...", flush=True)
            df = pd.read_excel(file_path)
            
            # Sampling acak yang representatif
            sample_df = df.sample(n=min(len(df), n_samples), random_state=42)
            for _, row in sample_df.iterrows():
                title = str(row.get('title', ''))
                raw_text = str(row.get('original', ''))
                if len(raw_text) < 50:
                    raw_text = str(row.get('cleaned', title))
                    
                cleaned = clean_text_id(raw_text)
                if len(cleaned) >= 40:
                    cat = categorize_text(title, raw_text)
                    valid_list.append({
                        'title': title,
                        'text': cleaned,
                        'label': 0,
                        'category': cat,
                        'source': source_name
                    })

    # Tambahkan hasil scraper live kesehatan jika ada
    scraped_cnn = r"dataset_kesehatan\RAW\dataset_cnn_raw.xlsx"
    if os.path.exists(scraped_cnn):
        df_c = pd.read_excel(scraped_cnn)
        for _, row in df_c.iterrows():
            title = str(row.get('Title', ''))
            raw_text = str(row.get('FullText', ''))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                valid_list.append({
                    'title': title,
                    'text': cleaned,
                    'label': 0,
                    'category': 'Kesehatan',
                    'source': 'CNN-Health-Live'
                })
                
    scraped_kompas = r"dataset_kesehatan\RAW\dataset_kompas_raw.xlsx"
    if os.path.exists(scraped_kompas):
        df_k = pd.read_excel(scraped_kompas)
        for _, row in df_k.iterrows():
            title = str(row.get('Title', ''))
            raw_text = str(row.get('FullText', ''))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                valid_list.append({
                    'title': title,
                    'text': cleaned,
                    'label': 0,
                    'category': 'Kesehatan',
                    'source': 'Kompas-Health-Live'
                })

    df_valid = pd.DataFrame(valid_list)
    df_valid = df_valid.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"  -> Total kandidat berita asli unik terkumpul: {len(df_valid)} baris", flush=True)
    
    if len(df_valid) > target_count:
        df_valid = df_valid.sample(n=target_count, random_state=42).reset_index(drop=True)
        
    print(f"  [OK] Berhasil mengamankan tepat {len(df_valid)} data berita asli (Label: 0)\n", flush=True)
    return df_valid


def main():
    output_dir = "dataset_multitopic_10k"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65, flush=True)
    print("     PEMBUATAN DATASET MULTI-TOPIK 10.000 DATA (BALANCED 50:50)", flush=True)
    print("=" * 65, flush=True)
    
    # 1. Ekstrak 5.000 hoaks
    df_hoax = extract_hoax_data(target_count=5000)
    
    # 2. Ekstrak 5.000 fakta
    df_valid = extract_valid_data(target_count=5000)
    
    # 3. Gabungkan dan Acak (Shuffle)
    print("[3/3] Menggabungkan, mengacak, dan membagi Train/Val/Test...", flush=True)
    full_df = pd.concat([df_hoax, df_valid], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 4. Stratified Split: 70% Train (7.000), 15% Val (1.500), 15% Test (1.500)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.30, random_state=42, stratify=full_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    # Simpan file lengkap
    combined_csv = os.path.join(output_dir, "dataset_multitopic_10k_combined.csv")
    combined_xlsx = os.path.join(output_dir, "dataset_multitopic_10k_combined.xlsx")
    full_df.to_csv(combined_csv, index=False)
    full_df.to_excel(combined_xlsx, index=False)
    
    # Simpan split
    train_csv = os.path.join(output_dir, "train_10k.csv")
    val_csv = os.path.join(output_dir, "val_10k.csv")
    test_csv = os.path.join(output_dir, "test_10k.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Buat salinan ke data_splits agar kompatibel langsung dengan trainer yang sudah ada
    os.makedirs("data_splits", exist_ok=True)
    train_df[['text', 'label', 'source']].to_csv(r"data_splits\train_multitopic_10k.csv", index=False)
    val_df[['text', 'label', 'source']].to_csv(r"data_splits\val_multitopic_10k.csv", index=False)
    test_df[['text', 'label', 'source']].to_csv(r"data_splits\test_multitopic_10k.csv", index=False)
    
    print("\n" + "=" * 65, flush=True)
    print("                 LAPORAN AKHIR DATASET 10.000", flush=True)
    print("=" * 65, flush=True)
    print(f" Total Seluruh Baris     : {len(full_df):,}")
    print(f"   - Fakta / Valid (0)   : {(full_df['label'] == 0).sum():,} ({((full_df['label'] == 0).sum()/len(full_df)*100):.1f}%)")
    print(f"   - Hoaks (1)           : {(full_df['label'] == 1).sum():,} ({((full_df['label'] == 1).sum()/len(full_df)*100):.1f}%)")
    
    print("\n Distribusi Topik:")
    cat_counts = full_df['category'].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(full_df) * 100
        print(f"   - {cat:<20}: {count:>5,} ({pct:>5.1f}%)")
        
    print("\n Distribusi Sumber Data:")
    src_counts = full_df['source'].value_counts()
    for src, count in src_counts.items():
        print(f"   - {src:<20}: {count:>5,}")
        
    print("\n Pembagian Data (Splits):")
    print(f"   - Train (70%)         : {len(train_df):,} baris -> {train_csv}")
    print(f"   - Validation (15%)    : {len(val_df):,} baris -> {val_csv}")
    print(f"   - Test (15%)          : {len(test_df):,} baris -> {test_csv}")
    print("\n Juga disalin ke data_splits/ untuk training:")
    print(r"   - data_splits\train_multitopic_10k.csv")
    print(r"   - data_splits\val_multitopic_10k.csv")
    print(r"   - data_splits\test_multitopic_10k.csv")
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    main()

