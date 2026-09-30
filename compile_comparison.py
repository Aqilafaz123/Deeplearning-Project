import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

svm_path = "results/svm/test_metrics.json"
cnn_path = "results/text_cnn/test_metrics.json"
bert_path = "results/indobert/test_metrics.json"

with open(svm_path, "r", encoding="utf-8") as f:
    svm_m = json.load(f)
with open(cnn_path, "r", encoding="utf-8") as f:
    cnn_m = json.load(f)
with open(bert_path, "r", encoding="utf-8") as f:
    bert_m = json.load(f)

data = [
    {
        "Arsitektur": "SVM + TF-IDF",
        "Accuracy": svm_m["accuracy"],
        "Precision (Macro)": svm_m["precision_macro"],
        "Recall (Macro)": svm_m["recall_macro"],
        "F1-Score (Macro)": svm_m["f1_macro"],
        "F1-Score (Weighted)": svm_m["f1_weighted"],
        "ROC-AUC": svm_m["roc_auc"]
    },
    {
        "Arsitektur": "TextCNN (Yoon Kim 2014)",
        "Accuracy": cnn_m["accuracy"],
        "Precision (Macro)": cnn_m["precision_macro"],
        "Recall (Macro)": cnn_m["recall_macro"],
        "F1-Score (Macro)": cnn_m["f1_macro"],
        "F1-Score (Weighted)": cnn_m["f1_weighted"],
        "ROC-AUC": cnn_m["roc_auc"]
    },
    {
        "Arsitektur": "IndoBERT (Fine-Tuned)",
        "Accuracy": bert_m["accuracy"],
        "Precision (Macro)": bert_m["precision_macro"],
        "Recall (Macro)": bert_m["recall_macro"],
        "F1-Score (Macro)": bert_m["f1_macro"],
        "F1-Score (Weighted)": bert_m["f1_weighted"],
        "ROC-AUC": bert_m["roc_auc"]
    }
]

df = pd.DataFrame(data)
os.makedirs("results", exist_ok=True)

csv_out = "results/comparison_table.csv"
md_out = "results/comparison_table.md"
df.to_csv(csv_out, index=False)

# Format markdown table
header = "| " + " | ".join(df.columns) + " |\n"
separator = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
rows = ""
for _, r in df.iterrows():
    row_vals = []
    for c in df.columns:
        val = r[c]
        if isinstance(val, float):
            row_vals.append(f"{val:.4f}")
        else:
            row_vals.append(str(val))
    rows += "| " + " | ".join(row_vals) + " |\n"

with open(md_out, "w", encoding="utf-8") as f:
    f.write("# Tabel Hasil Komparasi 3 Arsitektur Klasifikasi Berita Hoaks Politik\n\n")
    f.write(header + separator + rows)
    f.write("\n")

print("=== FINAL COMPARISON SUMMARY ===")
print(df.to_string(index=False))

# Plot high-res comparison chart
melted = pd.melt(
    df,
    id_vars=['Arsitektur'],
    value_vars=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    var_name='Metrik',
    value_name='Skor'
)

plt.figure(figsize=(11, 5.5), dpi=300)
sns.set_theme(style="whitegrid")
palette = ["#2b5c8f", "#d9534f", "#2ca02c"]

ax = sns.barplot(data=melted, x='Metrik', y='Skor', hue='Arsitektur', palette=palette)
plt.title("Perbandingan Kinerja: SVM+TF-IDF vs. TextCNN vs. IndoBERT", fontsize=14, fontweight='bold', pad=15)
plt.ylim(0.96, 1.005)
plt.ylabel("Skor", fontsize=12)
plt.xlabel("")
plt.legend(title="Arsitektur Model", fontsize=10, title_fontsize=11, loc='lower right')

for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8, color='black',
                    xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
barchart_path = "results/model_comparison_barchart.png"
plt.savefig(barchart_path)
plt.close()
print(f"Chart saved to: {barchart_path}")
