"""
Script Pengecekan Lengkap Seluruh Dataset Projek (Multi-Media & Multi-Topik)
Aman untuk Windows Console (Tanpa emoji unicode yang menyebabkan error encoding).
"""

import os
import sys
import pandas as pd

# Pastikan UTF-8 atau aman di konsol Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass


def inspect_excel_file(path, label_col="hoax"):
    try:
        df = pd.read_excel(path)
        lbl_col = label_col if label_col in df.columns else ('label' if 'label' in df.columns else None)
        dist = df[lbl_col].value_counts().to_dict() if lbl_col else "Tanpa Label"
        return len(df), dist, df
    except Exception as e:
        return 0, f"Error: {e}", None


def inspect_csv_file(path, label_col="label"):
    try:
        df = pd.read_csv(path)
        lbl_col = label_col if label_col in df.columns else ('hoax' if 'hoax' in df.columns else None)
        dist = df[lbl_col].value_counts().to_dict() if lbl_col else "Tanpa Label"
        return len(df), dist, df
    except Exception as e:
        return 0, f"Error: {e}", None


def main():
    print("\n" + "=" * 75)
    print("                PENGECEKAN LENGKAP SELURUH DATASET PROJEK")
    print("=" * 75)

    # 1. Dataset Hasil Scraping Terbaru (dataset_multimedia/RAW)
    raw_dir = r"dataset_multimedia\RAW"
    print(f"\n[1] HASIL SCRAPING TERBARU ({raw_dir}):")
    print("-" * 75)
    if os.path.exists(raw_dir):
        total_scraped_hoax = 0
        total_scraped_fakta = 0
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith('.xlsx') and "combined" not in f:
                path = os.path.join(raw_dir, f)
                n, dist, _ = inspect_excel_file(path)
                if isinstance(dist, dict):
                    total_scraped_hoax += dist.get(1, 0)
                    total_scraped_fakta += dist.get(0, 0)
                print(f"   * {f:<32} : {n:>5,} baris | Label: {dist}")
        print(f"\n   -> Subtotal Live Scraping : {total_scraped_hoax + total_scraped_fakta:,} artikel")
        print(f"      - Hoaks (1)           : {total_scraped_hoax:,}")
        print(f"      - Fakta / Asli (0)    : {total_scraped_fakta:,}")
    else:
        print("   (Folder belum ada)")

    # 2. Dataset Gabungan Multi-Media 12.000 (dataset_multimedia_12k)
    mm_dir = "dataset_multimedia_12k"
    print(f"\n[2] DATASET MULTI-MEDIA 12.000 SIAP TRAINING ({mm_dir}):")
    print("-" * 75)
    comb_12k = os.path.join(mm_dir, "dataset_multimedia_12k_combined.xlsx")
    if os.path.exists(comb_12k):
        n, dist, df_12k = inspect_excel_file(comb_12k, label_col="label")
        print(f"   * Total File Gabungan     : {n:,} baris | Distribusi: {dist}")
        for split in ["train_12k.csv", "val_12k.csv", "test_12k.csv"]:
            sp_path = os.path.join(mm_dir, split)
            if os.path.exists(sp_path):
                sn, sdist, _ = inspect_csv_file(sp_path)
                print(f"     - {split:<21} : {sn:>5,} baris | {sdist}")

        if df_12k is not None and "category" in df_12k.columns:
            print("\n   * Sebaran Topik/Kategori (12.000):")
            for cat, cnt in df_12k['category'].value_counts().items():
                pct = cnt / len(df_12k) * 100
                print(f"     - {cat:<24} : {cnt:>5,} artikel ({pct:>4.1f}%)")

        if df_12k is not None and "source" in df_12k.columns:
            print("\n   * Sebaran Media Sumber (12.000):")
            for src, cnt in df_12k['source'].value_counts().items():
                print(f"     - {src:<24} : {cnt:>5,} artikel")
    else:
        print("   (File belum ada)")

    # 3. Dataset Splits Siap Pakai untuk Model (data_splits)
    splits_dir = "data_splits"
    print(f"\n[3] DATA SPLITS AKTIF UNTUK TRAINING ({splits_dir}):")
    print("-" * 75)
    if os.path.exists(splits_dir):
        for f in sorted(os.listdir(splits_dir)):
            if f.endswith('.csv'):
                path = os.path.join(splits_dir, f)
                n, dist, _ = inspect_csv_file(path)
                print(f"   * {f:<32} : {n:>6,} baris | {dist}")
    else:
        print("   (Folder belum ada)")

    # 4. Arsip Data yang Tersedia di Projek (archive & komdigi)
    print(f"\n[4] KETERSEDIAAN DATA LAIN DI PROJEK ANDA:")
    print("-" * 75)
    komdigi_path = "komdigi_hoaks.csv"
    if os.path.exists(komdigi_path):
        n, _, _ = inspect_csv_file(komdigi_path)
        print(f"   * {komdigi_path:<32} : {n:>6,} baris (Arsip Hoaks Kominfo Resmi)")

    tbh_arch = r"archive\RAW\dataset_turnbackhoax_10k.xlsx"
    if os.path.exists(tbh_arch):
        n, _, _ = inspect_excel_file(tbh_arch)
        print(f"   * {tbh_arch:<32} : {n:>6,} baris (Arsip TurnBackHoax)")

    cnn_arch = r"archive\RAW\dataset_cnn_10k.xlsx"
    if os.path.exists(cnn_arch):
        n, _, _ = inspect_excel_file(cnn_arch)
        print(f"   * {cnn_arch:<32} : {n:>6,} baris (Arsip CNN Indonesia)")

    tempo_arch = r"archive\RAW\dataset_tempo_6k.xlsx"
    if os.path.exists(tempo_arch):
        n, _, _ = inspect_excel_file(tempo_arch)
        print(f"   * {tempo_arch:<32} : {n:>6,} baris (Arsip Tempo)")

    kompas_arch = r"archive\RAW\dataset_kompas_4k.xlsx"
    if os.path.exists(kompas_arch):
        n, _, _ = inspect_excel_file(kompas_arch)
        print(f"   * {kompas_arch:<32} : {n:>6,} baris (Arsip Kompas)")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
