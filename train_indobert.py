import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.data_loader import get_stratified_splits
from src.models.indobert_model import IndoBERTDataset, IndoBERTTrainer
from src.utils import (
    evaluate_predictions,
    print_metrics_summary,
    plot_and_save_confusion_matrix,
    plot_learning_curves,
    save_metrics_to_json
)


def run_indobert_experiment(base_dir="archive/Summarized", 
                            text_col="summarized", 
                            output_dir="results/indobert",
                            model_name="indobenchmark/indobert-base-p1",
                            epochs=3, 
                            batch_size=16, 
                            max_len=128,
                            lr=2e-5):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # 1. Load data splits
    train_df, val_df, test_df = get_stratified_splits(base_dir=base_dir, text_col=text_col)
    
    # 2. Instantiate Trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = IndoBERTTrainer(model_name=model_name, num_classes=2, lr=lr, device=device)
    
    # 3. Create datasets & DataLoaders
    train_dataset = IndoBERTDataset(train_df['text'].tolist(), train_df['label'].tolist(), trainer.tokenizer, max_len=max_len)
    val_dataset = IndoBERTDataset(val_df['text'].tolist(), val_df['label'].tolist(), trainer.tokenizer, max_len=max_len)
    test_dataset = IndoBERTDataset(test_df['text'].tolist(), test_df['label'].tolist(), trainer.tokenizer, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Train model
    save_dir = "saved_models/best_indobert"
    history = trainer.fit(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs, 
        save_dir=save_dir
    )
    
    # 5. Save Learning Curves
    plot_learning_curves(
        history['train_losses'],
        history['val_losses'],
        history['val_accs'],
        model_name="IndoBERT (Fine-Tuned)",
        save_path=os.path.join(output_dir, "learning_curves.png")
    )
    
    # 6. Evaluate on Test set
    test_loss, test_acc, test_preds, test_probs = trainer.evaluate(test_loader)
    test_metrics = evaluate_predictions(
        test_df['label'].tolist(),
        test_preds,
        test_probs,
        model_name="IndoBERT (Test)"
    )
    print_metrics_summary(test_metrics)
    
    save_metrics_to_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))
    plot_and_save_confusion_matrix(
        test_metrics['confusion_matrix'],
        labels=["Non-Hoax", "Hoax"],
        title="IndoBERT Confusion Matrix",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT")
    parser.add_argument("--base_dir", type=str, default="archive/Summarized")
    parser.add_argument("--text_col", type=str, default="summarized")
    parser.add_argument("--output_dir", type=str, default="results/indobert")
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobert-base-p1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    run_indobert_experiment(
        base_dir=args.base_dir,
        text_col=args.text_col,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr
    )
