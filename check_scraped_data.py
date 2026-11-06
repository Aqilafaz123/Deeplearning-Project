import os
import pandas as pd

raw_dir = r"dataset_multimedia\RAW"
print("=" * 65)
print("     DETAIL HASIL SCRAPING TERBARU (dataset_multimedia/RAW)")
print("=" * 65)

for f in sorted(os.listdir(raw_dir)):
    if f.endswith('.xlsx'):
        path = os.path.join(raw_dir, f)
        df = pd.read_excel(path)
        hoax_counts = df['hoax'].value_counts().to_dict() if 'hoax' in df.columns else {}
        print(f"File: {f:<38} | {len(df):>5} baris | Hoax: {hoax_counts}")

print("\n" + "=" * 65)
comb_path = os.path.join(raw_dir, 'dataset_multimedia_combined.xlsx')
if os.path.exists(comb_path):
    df_comb = pd.read_excel(comb_path)
    print("RINGKASAN FILE GABUNGAN (dataset_multimedia_combined.xlsx):")
    print(f"Total Baris      : {len(df_comb):,}")
    print(f"Fakta / Asli (0) : {(df_comb['hoax'] == 0).sum():,} ({((df_comb['hoax'] == 0).sum()/len(df_comb)*100):.1f}%)")
    print(f"Hoaks (1)        : {(df_comb['hoax'] == 1).sum():,} ({((df_comb['hoax'] == 1).sum()/len(df_comb)*100):.1f}%)")
    if 'Category' in df_comb.columns:
        print("\nSebaran Kategori:")
        for cat, cnt in df_comb['Category'].value_counts().items():
            print(f"  - {cat:<22}: {cnt} artikel")
print("=" * 65)

