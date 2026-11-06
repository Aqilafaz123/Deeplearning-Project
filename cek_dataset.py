"""
Script Cepat Cek Status Dataset Terbaru (dataset_multimedia/RAW)
Jalankan dengan perintah: python cek_dataset.py
"""

import os
import sys
import glob
from datetime import datetime
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

RAW_DIR = "dataset_multimedia/RAW"

def main():
    print("\n" + "=" * 72)
    print("             STATUS DATASET HASIL SCRAPING TERBARU")
    print("=" * 72)
    
    if not os.path.exists(RAW_DIR):
        print(f"[!] Folder '{RAW_DIR}' belum ditemukan.")
        return

    excel_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.xlsx")))
    
    media_data = []
    total_raw_rows = 0
    total_hoax = 0
    total_fakta = 0

    for fpath in excel_files:
        fname = os.path.basename(fpath)
        if "combined" in fname:
            continue
            
        try:
            df = pd.read_excel(fpath)
            rows = len(df)
            total_raw_rows += rows
            
            lbl_col = 'hoax' if 'hoax' in df.columns else ('label' if 'label' in df.columns else None)
            hoax_count = (df[lbl_col] == 1).sum() if lbl_col else 0
            fakta_count = (df[lbl_col] == 0).sum() if lbl_col else 0
            
            total_hoax += hoax_count
            total_fakta += fakta_count
            
            mtime = os.path.getmtime(fpath)
            update_time = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M:%S")
            
            media_data.append({
                "file": fname,
                "rows": rows,
                "hoax": hoax_count,
                "fakta": fakta_count,
                "updated": update_time
            })
        except Exception as e:
            print(f"  [!] Gagal membaca {fname}: {e}")

    print(f"{'Nama File':<32} | {'Baris':>7} | {'Label':<18} | {'Terakhir Update':<19}")
    print("-" * 72)
    for m in media_data:
        label_str = f"Hoaks: {m['hoax']:,}" if m['hoax'] > 0 else f"Fakta: {m['fakta']:,}"
        print(f"{m['file']:<32} | {m['rows']:>7,} | {label_str:<18} | {m['updated']:<19}")

    print("-" * 72)
    print(f" TOTAL DATA SEBELUM DEDUPLIKASI : {total_raw_rows:,} baris")
    print(f"   * Fakta / Asli (0)           : {total_fakta:,} artikel")
    print(f"   * Hoaks (1)                  : {total_hoax:,} artikel")

    # Cek file gabungan jika ada
    comb_csv = os.path.join(RAW_DIR, "dataset_multimedia_combined.csv")
    comb_xlsx = os.path.join(RAW_DIR, "dataset_multimedia_combined.xlsx")
    
    if os.path.exists(comb_csv):
        df_comb = pd.read_csv(comb_csv)
        print("\n" + "-" * 72)
        print(f" TOTAL DATASET BERSIH & UNIK    : {len(df_comb):,} artikel")
        if 'hoax' in df_comb.columns:
            hc = df_comb['hoax'].value_counts()
            print(f"   * Fakta / Asli (0)           : {hc.get(0, 0):,} artikel")
            print(f"   * Hoaks (1)                  : {hc.get(1, 0):,} artikel")
        print(f" File tersimpan di              : {comb_csv}")

    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()

