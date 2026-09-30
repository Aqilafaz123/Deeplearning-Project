import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_svm import run_svm_experiment
from train_cnn import run_cnn_experiment
from train_indobert import run_indobert_experiment


def plot_comparison_chart(df_results, save_path="results/model_comparison_barchart.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    melted = pd.melt(
        df_results,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        var_name='Metric',
        value_name='Score'
    )
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted")
    
    ax = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette=palette)
    plt.title("Comparative Performance Analysis: SVM vs TextCNN vs IndoBERT", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")
    plt.legend(title="Architecture", title_fontsize='11', fontsize='10', loc='lower right')
    
    # Annotate values on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 2),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Comparison] Bar chart saved to: {save_path}")


def run_all_experiments(base_dir="archive/Summarized", text_col="summarized", cnn_epochs=12, bert_epochs=3):
    print("\n=======================================================")
    print("      STARTING COMPARATIVE STUDY: 3 ARCHITECTURES      ")
    print("=======================================================\n")
    
    # 1. Train SVM + TF-IDF
    print(">>> [1/3] TRAINING MODEL 1: SVM + TF-IDF ...")
    svm_metrics = run_svm_experiment(base_dir=base_dir, text_col=text_col, output_dir="results/svm")
    
    # 2. Train TextCNN
    print("\n>>> [2/3] TRAINING MODEL 2: TextCNN (Yoon Kim 2014) ...")
    cnn_metrics = run_cnn_experiment(base_dir=base_dir, text_col=text_col, output_dir="results/text_cnn", epochs=cnn_epochs)
    
    # 3. Fine-Tune IndoBERT
    print("\n>>> [3/3] TRAINING MODEL 3: IndoBERT (indobenchmark/indobert-base-p1) ...")
    bert_metrics = run_indobert_experiment(base_dir=base_dir, text_col=text_col, output_dir="results/indobert", epochs=bert_epochs)
    
    # Compile comparison table
    summary_data = [
        {
            "Model": "SVM + TF-IDF",
            "Accuracy": svm_metrics["accuracy"],
            "Precision (Macro)": svm_metrics["precision_macro"],
            "Recall (Macro)": svm_metrics["recall_macro"],
            "F1-Score (Macro)": svm_metrics["f1_macro"],
            "F1-Score (Weighted)": svm_metrics["f1_weighted"],
            "ROC-AUC": svm_metrics.get("roc_auc")
        },
        {
            "Model": "TextCNN (Yoon Kim 2014)",
            "Accuracy": cnn_metrics["accuracy"],
            "Precision (Macro)": cnn_metrics["precision_macro"],
            "Recall (Macro)": cnn_metrics["recall_macro"],
            "F1-Score (Macro)": cnn_metrics["f1_macro"],
            "F1-Score (Weighted)": cnn_metrics["f1_weighted"],
            "ROC-AUC": cnn_metrics.get("roc_auc")
        },
        {
            "Model": "IndoBERT (Fine-Tuned)",
            "Accuracy": bert_metrics["accuracy"],
            "Precision (Macro)": bert_metrics["precision_macro"],
            "Recall (Macro)": bert_metrics["recall_macro"],
            "F1-Score (Macro)": bert_metrics["f1_macro"],
            "F1-Score (Weighted)": bert_metrics["f1_weighted"],
            "ROC-AUC": bert_metrics.get("roc_auc")
        }
    ]
    
    df_results = pd.DataFrame(summary_data)
    os.makedirs("results", exist_ok=True)
    
    csv_path = "results/comparison_table.csv"
    md_path = "results/comparison_table.md"
    df_results.to_csv(csv_path, index=False)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hasil Komparasi Performa 3 Arsitektur (Journal Ready)\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n")
        
    print("\n=======================================================")
    print("               FINAL COMPARISON SUMMARY                ")
    print("=======================================================")
    print(df_results.to_string(index=False))
    print("=======================================================\n")
    print(f"[Comparison] Results saved to:\n  - CSV: {csv_path}\n  - Markdown: {md_path}")
    
    # Plot comparison chart
    plot_comparison_chart(df_results, "results/model_comparison_barchart.png")
    
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete comparative training")
    parser.add_argument("--base_dir", type=str, default="archive/Summarized")
    parser.add_argument("--text_col", type=str, default="summarized")
    parser.add_argument("--cnn_epochs", type=int, default=12)
    parser.add_argument("--bert_epochs", type=int, default=3)
    args = parser.parse_args()
    
    run_all_experiments(
        base_dir=args.base_dir,
        text_col=args.text_col,
        cnn_epochs=args.cnn_epochs,
        bert_epochs=args.bert_epochs
    )
