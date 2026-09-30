import pandas as pd
import os

base = r"d:\deep learning\dataset_politics"
print("=== Cleaned Folder ===")
for f in os.listdir(os.path.join(base, "Cleaned")):
    if f.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(base, "Cleaned", f))
        col = 'Clean Narasi' if 'Clean Narasi' in df.columns else 'text_new'
        print(f"{f}: total={len(df)}, text nulls={df[col].isna().sum()}, hoax nulls={df['hoax'].isna().sum()}")

print("\n=== Summarized Folder ===")
for f in os.listdir(os.path.join(base, "Summarized")):
    if f.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(base, "Summarized", f))
        print(f"{f}: total={len(df)}, cleaned nulls={df['cleaned'].isna().sum()}, summarized nulls={df['summarized'].isna().sum()}, label nulls={df['label'].isna().sum()}")
