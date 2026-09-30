import argparse
import os
import pickle
import torch
from torch.utils.data import DataLoader

from src.data_loader import get_stratified_splits
from src.text_preprocessing import TextVocabulary
from src.models.text_cnn import YoonKimTextCNN, TextCNNDataset, TextCNNTrainer
from src.utils import (
    evaluate_predictions,
    print_metrics_summary,
    plot_and_save_confusion_matrix,
    plot_learning_curves,
    save_metrics_to_json
)


def run_cnn_experiment(base_dir="archive/Summarized", 
                       text_col="summarized", 
                       output_dir="results/text_cnn",
                       epochs=15, 
                       batch_size=32, 
                       embed_dim=128, 
                       max_len=256,
                       lr=1e-3):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # 1. Load data splits
    train_df, val_df, test_df = get_stratified_splits(base_dir=base_dir, text_col=text_col)
    
    # 2. Build and save vocabulary
    vocab = TextVocabulary()
    vocab.build_vocab(train_df['text'].tolist(), max_vocab_size=30000, min_freq=2)
    vocab_path = "saved_models/text_cnn_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"[TextCNN] Vocabulary saved to {vocab_path}")
    
    # 3. Create datasets & DataLoaders
    train_dataset = TextCNNDataset(train_df['text'].tolist(), train_df['label'].tolist(), vocab, max_len=max_len)
    val_dataset = TextCNNDataset(val_df['text'].tolist(), val_df['label'].tolist(), vocab, max_len=max_len)
    test_dataset = TextCNNDataset(test_df['text'].tolist(), test_df['label'].tolist(), vocab, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Instantiate Model & Trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoonKimTextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        num_classes=2,
        dropout=0.5,
        pad_idx=vocab.pad_idx
    )
    
    trainer = TextCNNTrainer(model=model, device=device, lr=lr)
    
    # 5. Train model
    cnn_weights_path = "saved_models/best_text_cnn.pt"
    history = trainer.fit(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs, 
        patience=4, 
        save_path=cnn_weights_path
    )
    
    # 6. Save Learning Curves
    plot_learning_curves(
        history['train_losses'],
        history['val_losses'],
        history['val_accs'],
        model_name="TextCNN (Yoon Kim 2014)",
        save_path=os.path.join(output_dir, "learning_curves.png")
    )
    
    # 7. Evaluate on Test set
    test_loss, test_acc, test_preds, test_probs = trainer.evaluate(test_loader)
    test_metrics = evaluate_predictions(
        test_df['label'].tolist(),
        test_preds,
        test_probs,
        model_name="TextCNN (Test)"
    )
    print_metrics_summary(test_metrics)
    
    save_metrics_to_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))
    plot_and_save_confusion_matrix(
        test_metrics['confusion_matrix'],
        labels=["Non-Hoax", "Hoax"],
        title="TextCNN Confusion Matrix",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TextCNN")
    parser.add_argument("--base_dir", type=str, default="archive/Summarized")
    parser.add_argument("--text_col", type=str, default="summarized")
    parser.add_argument("--output_dir", type=str, default="results/text_cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    run_cnn_experiment(
        base_dir=args.base_dir,
        text_col=args.text_col,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        max_len=args.max_len,
        lr=args.lr
    )
