"""
Script Utama Pengambil Dataset Berita (Scraper) Multi-Media & Multi-Topik
Mendukung:
  1. Media Hoaks (Label: 1):
     - TurnBackHoax (Mafindo)
     - Liputan6 Cek Fakta (Terakreditasi IFCN)
  2. Media Berita Asli / Fakta (Label: 0):
     - CNN Indonesia
     - Kompas.com
     - Antara News (Kantor Berita Nasional)

Fitur Khusus:
  - Smart Append: Otomatis menggabungkan data baru dengan data lama tanpa menimpa/menghapus.
  - Deduplikasi: Menghilangkan duplikat berbasis URL dan Judul artikel.
  - Safe Save: Mengatasi Windows PermissionError jika file sedang dibuka di Excel.
  - Pagination Control: Mendukung --start_page untuk mengambil halaman lama.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Pastikan output terminal tidak ter-buffer
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from src.scrapers import (
    scrape_turnbackhoax,
    scrape_liputan6_cekfakta,
    scrape_merdeka_cekfakta,
    scrape_detik,
    scrape_kompas,
    scrape_cnn,
    scrape_antara
)


def safe_save(df, target_path):
    """Menyimpan DataFrame ke disk dengan penanganan file lock di Windows."""
    try:
        if target_path.endswith('.xlsx'):
            df.to_excel(target_path, index=False)
        else:
            df.to_csv(target_path, index=False, encoding='utf-8')
        return target_path
    except PermissionError:
        base, ext = os.path.splitext(target_path)
        fallback = f"{base}_new{ext}"
        print(f"   [!] File '{os.path.basename(target_path)}' sedang dibuka di program lain (Excel/Editor).")
        print(f"   [!] Menyimpan ke file alternatif: '{os.path.basename(fallback)}'")
        if fallback.endswith('.xlsx'):
            df.to_excel(fallback, index=False)
        else:
            df.to_csv(fallback, index=False, encoding='utf-8')
        return fallback


def save_or_append_df(df_new, file_path, append=True, id_cols=["Url", "Title"]):
    """
    Menyimpan atau menambahkan data baru ke file dataset yang sudah ada.
    Melakukan deduplikasi otomatis agar tidak ada data ganda.
    """
    if df_new is None or df_new.empty:
        return pd.DataFrame()

    if append and os.path.exists(file_path):
        try:
            df_old = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            old_len = len(df_old)
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            
            # Deduplikasi berdasarkan Url atau Title
            subset = [c for c in id_cols if c in df_merged.columns]
            if subset:
                df_merged = df_merged.drop_duplicates(subset=subset).reset_index(drop=True)
                
            added = len(df_merged) - old_len
            saved_to = safe_save(df_merged, file_path)
            print(f"   [V] Tersimpan ke: {saved_to} (Sebelumnya: {old_len:,} | Tambahan: +{added:,} | Total Sekarang: {len(df_merged):,})", flush=True)
            return df_merged
        except Exception as e:
            print(f"   [!] Gagal membaca file lama ({e}), menyimpan data baru langsung...", flush=True)

    saved_to = safe_save(df_new, file_path)
    print(f"   [V] Tersimpan ke: {saved_to} (Total: {len(df_new):,} baris)", flush=True)
    return df_new


def main():
    parser = argparse.ArgumentParser(description="Scraper Dataset Berita Multi-Media (Hoaks & Fakta)")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "turnbackhoax", "liputan6", "merdeka", "detik", "cnn", "kompas", "antara"],
        help="Sumber media yang ingin diambil (default: all)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Kategori berita: 'all', 'kesehatan', 'bantuan', 'lowongan', 'tekno', 'ekonomi' (default: all)"
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=1,
        help="Halaman awal yang ingin diambil (default: 1)"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Jumlah halaman indeks per media (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Jeda antar request dalam detik (default: 0.3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_multimedia/RAW",
        help="Folder tujuan penyimpanan file (default: dataset_multimedia/RAW)"
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Jika diatur, file lama akan ditimpa alih-alih ditambahkan (append)"
    )
    args = parser.parse_args()
    do_append = not args.no_append

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70, flush=True)
    print("      SCRAPER DATASET MULTI-MEDIA INDONESIA (MULTI-TOPIK)", flush=True)
    print("=" * 70, flush=True)
    print(f" Target Sumber   : {args.source.upper()}", flush=True)
    print(f" Kategori        : {args.category.upper()}", flush=True)
    print(f" Rentang Halaman : Hal {args.start_page} s/d {args.start_page + args.pages - 1} ({args.pages} hal)", flush=True)
    print(f" Mode Append     : {'AKTIF (Menambah data lama)' if do_append else 'NONAKTIF (Timpa file)'}", flush=True)
    print(f" Jeda Request    : {args.delay} detik", flush=True)
    print(f" Folder Output   : {os.path.abspath(args.output_dir)}", flush=True)
    print("=" * 70, flush=True)

    # 1. TurnBackHoax (Label: 1 / Hoaks)
    if args.source in ["all", "turnbackhoax"]:
        df_tbh = scrape_turnbackhoax(category=args.category, max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_tbh.empty:
            tbh_path = os.path.join(args.output_dir, "dataset_turnbackhoax_raw.xlsx")
            save_or_append_df(df_tbh, tbh_path, append=do_append)

    # 2. Liputan6 Cek Fakta (Label: 1 / Hoaks)
    if args.source in ["all", "liputan6"]:
        df_lip = scrape_liputan6_cekfakta(max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_lip.empty:
            lip_path = os.path.join(args.output_dir, "dataset_liputan6_raw.xlsx")
            save_or_append_df(df_lip, lip_path, append=do_append)

    # 3. Merdeka Cek Fakta (Label: 1 / Hoaks)
    if args.source in ["all", "merdeka"]:
        df_merdeka = scrape_merdeka_cekfakta(max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_merdeka.empty:
            merdeka_path = os.path.join(args.output_dir, "dataset_merdeka_raw.xlsx")
            save_or_append_df(df_merdeka, merdeka_path, append=do_append)

    # 4. Detik.com (Label: 0 / Fakta)
    if args.source in ["all", "detik"]:
        df_detik = scrape_detik(category=args.category, max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_detik.empty:
            detik_path = os.path.join(args.output_dir, "dataset_detik_raw.xlsx")
            save_or_append_df(df_detik, detik_path, append=do_append)

    # 5. CNN Indonesia (Label: 0 / Fakta)
    if args.source in ["all", "cnn"]:
        df_cnn = scrape_cnn(category=args.category, max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_cnn.empty:
            cnn_path = os.path.join(args.output_dir, "dataset_cnn_raw.xlsx")
            save_or_append_df(df_cnn, cnn_path, append=do_append)

    # 6. Kompas.com (Label: 0 / Fakta)
    if args.source in ["all", "kompas"]:
        df_kompas = scrape_kompas(category=args.category, max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_kompas.empty:
            kompas_path = os.path.join(args.output_dir, "dataset_kompas_raw.xlsx")
            save_or_append_df(df_kompas, kompas_path, append=do_append)

    # 7. Antara News (Label: 0 / Fakta)
    if args.source in ["all", "antara"]:
        df_antara = scrape_antara(category=args.category, max_pages=args.pages, delay=args.delay, start_page=args.start_page)
        if not df_antara.empty:
            antara_path = os.path.join(args.output_dir, "dataset_antara_raw.xlsx")
            save_or_append_df(df_antara, antara_path, append=do_append)

    # Gabungkan dan rekapitulasi seluruh file di folder output
    print("\n" + "=" * 70, flush=True)
    print("            REKAPITULASI SELURUH DATASET MULTIMEDIA", flush=True)
    print("=" * 70, flush=True)
    
    all_raw_files = [
        ("dataset_turnbackhoax_raw.xlsx", "TurnBackHoax", 1),
        ("dataset_liputan6_raw.xlsx", "Liputan6", 1),
        ("dataset_merdeka_raw.xlsx", "Merdeka", 1),
        ("dataset_detik_raw.xlsx", "Detik", 0),
        ("dataset_kompas_raw.xlsx", "Kompas", 0),
        ("dataset_antara_raw.xlsx", "Antara", 0),
        ("dataset_cnn_raw.xlsx", "CNN", 0),
    ]
    
    combined_dfs = []
    for fname, source, label in all_raw_files:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            try:
                df_item = pd.read_excel(fpath)
                df_item['Source'] = source
                if 'hoax' not in df_item.columns:
                    df_item['hoax'] = label
                print(f"   * {source:<15}: {len(df_item):,} artikel ({'Hoaks' if label==1 else 'Fakta'})", flush=True)
                combined_dfs.append(df_item)
            except Exception as e:
                print(f"   * {source:<15}: Error membaca ({e})", flush=True)

    if combined_dfs:
        merged_all = pd.concat(combined_dfs, ignore_index=True)
        before_dedup = len(merged_all)
        subset_cols = [c for c in ['Url', 'Title'] if c in merged_all.columns]
        if subset_cols:
            merged_all = merged_all.drop_duplicates(subset=subset_cols).reset_index(drop=True)
            
        combined_xlsx = os.path.join(args.output_dir, "dataset_multimedia_combined.xlsx")
        combined_csv = os.path.join(args.output_dir, "dataset_multimedia_combined.csv")
        safe_save(merged_all, combined_xlsx)
        safe_save(merged_all, combined_csv)
        
        print("-" * 70, flush=True)
        print(f" TOTAL DATA KESELURUHAN SETELAH DEDUPLIKASI: {len(merged_all):,} artikel", flush=True)
        counts = merged_all["hoax"].value_counts()
        print(f"   - Fakta / Asli (0) : {counts.get(0, 0):,} artikel", flush=True)
        print(f"   - Hoaks (1)        : {counts.get(1, 0):,} artikel", flush=True)
        print(f" File gabungan tersimpan di:", flush=True)
        print(f"   - {combined_xlsx}", flush=True)
        print(f"   - {combined_csv}", flush=True)
        print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
