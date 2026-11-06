"""
Script Generator Dataset Multi-Media 12.000 Data (Prioritas 100% Data Live Scraping 3.995)
Menangani file lock Windows secara aman.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.text_preprocessing import clean_text_id

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def safe_save_df(df, filepath, is_excel=False):
    """Menyimpan dataframe secara aman, jika file sedang dibuka Excel, simpan dengan nama alternatif."""
    try:
        if is_excel:
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        return filepath
    except PermissionError:
        base, ext = os.path.splitext(filepath)
        alt_path = f"{base}_updated{ext}"
        print(f"  [!] Catatan: '{filepath}' sedang dibuka di Excel/program lain. Disimpan ke: '{alt_path}'", flush=True)
        if is_excel:
            df.to_excel(alt_path, index=False)
        else:
            df.to_csv(alt_path, index=False)
        return alt_path


def categorize_text(title, text=""):
    content = f"{str(title)} {str(text)}".lower()
    if any(k in content for k in ['kesehatan', 'obat', 'penyakit', 'vaksin', 'covid', 'corona', 'dokter', 'kanker', 'diabetes', 'darah', 'rumah sakit', 'virus', 'stroke', 'gula', 'imunisasi', 'gizi', 'diet', 'medis', 'bpjs', 'vitamin', 'gejala']):
        return "Kesehatan"
    if any(k in content for k in ['gempa', 'tsunami', 'gunung', 'banjir', 'bencana', 'erupsi', 'bmkg', 'longsor', 'cuaca ekstrem', 'puting beliung', 'karhutla']):
        return "Bencana Alam"
    if any(k in content for k in ['penipuan', 'bantuan', 'dana', 'hadiah', 'undian', 'giveaway', 'pulsa', 'link', 'tautan', 'blt', 'bansos', 'pinjol', 'investasi bodong', 'saldo']):
        return "Penipuan & Bansos"
    if any(k in content for k in ['lowongan', 'loker', 'rekrutmen', 'cpns', 'bumn', 'buka rekrutmen', 'karir']):
        return "Lowongan Kerja"
    if any(k in content for k in ['ekonomi', 'bisnis', 'inflasi', 'saham', 'rupiah', 'ihsg', 'ekspor', 'impor', 'subsidi', 'bbm', 'umkm', 'pasar']):
        return "Ekonomi & Bisnis"
    if any(k in content for k in ['teknologi', 'tekno', 'ai', 'gadget', 'sains', 'edukasi', 'pendidikan', 'kampus', 'sekolah', 'riset']):
        return "Sains, Tekno & Edukasi"
    if any(k in content for k in ['presiden', 'prabowo', 'jokowi', 'ganjar', 'anies', 'gibran', 'dpr', 'kpu', 'pemilu', 'pilpres', 'pilkada', 'partai', 'pdi', 'gerindra', 'golkar', 'nasdem', 'menteri', 'bawaslu']):
        return "Politik"
    return "Sosial & Umum"


def load_live_scraped_data():
    raw_dir = r"dataset_multimedia\RAW"
    files = [
        ('dataset_turnbackhoax_raw.xlsx', 1, 'TurnBackHoax (Live)'),
        ('dataset_liputan6_raw.xlsx', 1, 'Liputan6 (Live)'),
        ('dataset_kompas_raw.xlsx', 0, 'Kompas.com (Live)'),
        ('dataset_antara_raw.xlsx', 0, 'Antara News (Live)'),
        ('dataset_cnn_raw.xlsx', 0, 'CNN Indonesia (Live)')
    ]
    
    records = []
    print("\n[1/4] Memuat seluruh 3.995 data hasil Live Scraping terbaru...", flush=True)
    for fname, lbl, src in files:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_excel(fpath)
            cnt = 0
            for _, row in df.iterrows():
                title = str(row.get('Title', ''))
                raw_text = str(row.get('FullText', ''))
                cleaned = clean_text_id(raw_text)
                if len(cleaned) >= 40:
                    records.append({
                        'title': title,
                        'text': cleaned,
                        'label': lbl,
                        'source': src,
                        'category': categorize_text(title, raw_text)
                    })
                    cnt += 1
            print(f"  -> {fname:<30}: {cnt:>5,} artikel dimuat ({'Hoaks' if lbl==1 else 'Fakta'})", flush=True)
            
    df_live = pd.DataFrame(records).drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"  [OK] Total Live Scraped Unik: {len(df_live):,} artikel (Hoaks: {(df_live['label']==1).sum():,}, Fakta: {(df_live['label']==0).sum():,})\n", flush=True)
    return df_live


def main():
    target_per_class = 6000  # Total 12.000 data seimbang
    output_dir = "dataset_multimedia_12k"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("   INTEGRASI DATASET MULTI-MEDIA 12.000 (MEMASUKKAN SEMUA DATA LIVE)")
    print("=" * 70, flush=True)
    
    # 1. Muat semua data live scraping
    df_live = load_live_scraped_data()
    live_hoax = df_live[df_live['label'] == 1].copy()
    live_valid = df_live[df_live['label'] == 0].copy()
    
    # 2. Lengkapi Hoaks hingga 6.000
    needed_hoax = target_per_class - len(live_hoax)
    print(f"[2/4] Menggenapkan data Hoaks (Sudah ada {len(live_hoax):,} live, butuh {needed_hoax:,} lagi)...", flush=True)
    supp_hoax = []
    
    if os.path.exists("komdigi_hoaks.csv"):
        df_k = pd.read_csv("komdigi_hoaks.csv")
        for _, row in df_k.iterrows():
            title = str(row.get('title', ''))
            raw_text = str(row.get('body_text', ''))
            if len(raw_text) < 40:
                raw_text = str(row.get('excerpt', title))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                supp_hoax.append({
                    'title': title,
                    'text': cleaned,
                    'label': 1,
                    'source': 'Kominfo/Komdigi',
                    'category': categorize_text(title, raw_text)
                })
                
    tbh_arch = r"archive\Summarized\dataset_turnbackhoax_summarized.xlsx"
    if os.path.exists(tbh_arch):
        df_t = pd.read_excel(tbh_arch)
        sample = df_t.sample(n=min(len(df_t), 4000), random_state=42)
        for _, row in sample.iterrows():
            title = str(row.get('title', ''))
            raw_text = str(row.get('original', row.get('cleaned', title)))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                supp_hoax.append({
                    'title': title,
                    'text': cleaned,
                    'label': 1,
                    'source': 'TurnBackHoax (Arsip)',
                    'category': categorize_text(title, raw_text)
                })
                
    df_supp_h = pd.DataFrame(supp_hoax).drop_duplicates(subset=['title']).reset_index(drop=True)
    df_supp_h = df_supp_h[~df_supp_h['title'].isin(live_hoax['title'])].reset_index(drop=True)
    if len(df_supp_h) > needed_hoax:
        df_supp_h = df_supp_h.sample(n=needed_hoax, random_state=42).reset_index(drop=True)
        
    full_hoax = pd.concat([live_hoax, df_supp_h], ignore_index=True)
    print(f"  [OK] Data Hoaks Lengkap: {len(full_hoax):,} baris (Live: {len(live_hoax):,}, Pelengkap: {len(df_supp_h):,})\n", flush=True)

    # 3. Lengkapi Fakta hingga 6.000
    needed_valid = target_per_class - len(live_valid)
    print(f"[3/4] Menggenapkan data Fakta (Sudah ada {len(live_valid):,} live, butuh {needed_valid:,} lagi)...", flush=True)
    supp_valid = []
    
    cnn_arch = r"archive\Summarized\dataset_cnn_summarized.xlsx"
    if os.path.exists(cnn_arch):
        df_c = pd.read_excel(cnn_arch).sample(n=min(3000, needed_valid), random_state=42)
        for _, row in df_c.iterrows():
            title = str(row.get('title', ''))
            raw_text = str(row.get('original', row.get('cleaned', title)))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                supp_valid.append({
                    'title': title,
                    'text': cleaned,
                    'label': 0,
                    'source': 'CNN Indonesia (Arsip)',
                    'category': categorize_text(title, raw_text)
                })
                
    tempo_arch = r"archive\Summarized\dataset_tempo_summarized.xlsx"
    if os.path.exists(tempo_arch):
        df_tmp = pd.read_excel(tempo_arch).sample(n=min(3000, needed_valid), random_state=42)
        for _, row in df_tmp.iterrows():
            title = str(row.get('title', ''))
            raw_text = str(row.get('original', row.get('cleaned', title)))
            cleaned = clean_text_id(raw_text)
            if len(cleaned) >= 40:
                supp_valid.append({
                    'title': title,
                    'text': cleaned,
                    'label': 0,
                    'source': 'Tempo.co (Arsip)',
                    'category': categorize_text(title, raw_text)
                })

    df_supp_v = pd.DataFrame(supp_valid).drop_duplicates(subset=['title']).reset_index(drop=True)
    df_supp_v = df_supp_v[~df_supp_v['title'].isin(live_valid['title'])].reset_index(drop=True)
    if len(df_supp_v) > needed_valid:
        df_supp_v = df_supp_v.sample(n=needed_valid, random_state=42).reset_index(drop=True)
        
    full_valid = pd.concat([live_valid, df_supp_v], ignore_index=True)
    print(f"  [OK] Data Fakta Lengkap: {len(full_valid):,} baris (Live: {len(live_valid):,}, Pelengkap: {len(df_supp_v):,})\n", flush=True)

    # 4. Gabungkan dan Split
    print("[4/4] Menggabungkan, mengacak (shuffle), dan membagi Train/Val/Test...", flush=True)
    full_df = pd.concat([full_hoax, full_valid], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 70% Train (8.400), 15% Val (1.800), 15% Test (1.800)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.30, random_state=42, stratify=full_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    # Simpan aman
    safe_save_df(full_df, os.path.join(output_dir, "dataset_multimedia_12k_combined.csv"))
    safe_save_df(full_df, os.path.join(output_dir, "dataset_multimedia_12k_combined.xlsx"), is_excel=True)
    
    safe_save_df(train_df, os.path.join(output_dir, "train_12k.csv"))
    safe_save_df(val_df, os.path.join(output_dir, "val_12k.csv"))
    safe_save_df(test_df, os.path.join(output_dir, "test_12k.csv"))
    
    # Salin ke data_splits
    os.makedirs("data_splits", exist_ok=True)
    safe_save_df(train_df[['text', 'label', 'source']], r"data_splits\train_multimedia_12k.csv")
    safe_save_df(val_df[['text', 'label', 'source']], r"data_splits\val_multimedia_12k.csv")
    safe_save_df(test_df[['text', 'label', 'source']], r"data_splits\test_multimedia_12k.csv")
    
    print("\n" + "=" * 70, flush=True)
    print("       LAPORAN AKHIR: DATASET 12.000 LENGKAP DENGAN 3.995 LIVE SCRAPED")
    print("=" * 70, flush=True)
    print(f" Total Baris Dataset    : {len(full_df):,}")
    print(f"   - Fakta / Asli (0)   : {(full_df['label']==0).sum():,} (50.0%)")
    print(f"   - Hoaks (1)          : {(full_df['label']==1).sum():,} (50.0%)")
    
    print("\n Kontribusi Seluruh Media (Termasuk Live Scraping Anda):")
    for src, cnt in full_df['source'].value_counts().items():
        pct = cnt / len(full_df) * 100
        print(f"   * {src:<26}: {cnt:>5,} artikel ({pct:>4.1f}%)")
        
    print("\n Sebaran Kategori Topik:")
    for cat, cnt in full_df['category'].value_counts().items():
        pct = cnt / len(full_df) * 100
        print(f"   * {cat:<26}: {cnt:>5,} artikel ({pct:>4.1f}%)")
        
    print("\n Pembagian Set:")
    print(f"   * Train (70%)        : {len(train_df):,} baris -> data_splits\\train_multimedia_12k.csv")
    print(f"   * Validation (15%)   : {len(val_df):,} baris -> data_splits\\val_multimedia_12k.csv")
    print(f"   * Test (15%)         : {len(test_df):,} baris -> data_splits\\test_multimedia_12k.csv")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
