import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_predictions(y_true, y_pred, y_probs=None, model_name="Model"):
    """
    Computes comprehensive classification metrics suitable for academic journal publication.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    if y_probs is not None:
        try:
            if len(np.shape(y_probs)) == 2 and y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_probs)
            metrics["roc_auc"] = float(auc)
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def print_metrics_summary(metrics):
    """
    Prints a formatted summary table for console output.
    """
    print(f"\n========================================================")
    print(f"       EVALUATION RESULTS: {metrics['model_name']}")
    print(f"========================================================")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro)  : {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro)     : {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro)   : {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"--------------------------------------------------------")
    print("Confusion Matrix:")
    for row in metrics['confusion_matrix']:
        print(f"  {row}")
    print(f"========================================================\n")


def plot_and_save_confusion_matrix(cm, labels=["Non-Hoax", "Hoax"], title="Confusion Matrix", save_path="confusion_matrix.png"):
    """
    Plots and saves a high-resolution confusion matrix heatmap for journal figures.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Utils] Confusion matrix figure saved to: {save_path}")


def plot_learning_curves(train_losses, val_losses, val_accs, model_name="Model", save_path="learning_curves.png"):
    """
    Plots loss and validation accuracy curves over epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4)
    ax1.set_title(f'{model_name} - Loss per Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    ax2.plot(epochs, val_accs, 'g-^', label='Val Accuracy', markersize=4)
    ax2.set_title(f'{model_name} - Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Utils] Learning curves saved to: {save_path}")


def save_metrics_to_json(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"[Utils] Metrics saved to: {save_path}")
