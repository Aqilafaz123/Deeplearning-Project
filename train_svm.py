import argparse
import os
from src.data_loader import get_stratified_splits
from src.models.svm_model import SVMTextClassifier
from src.utils import (
    evaluate_predictions,
    print_metrics_summary,
    plot_and_save_confusion_matrix,
    save_metrics_to_json
)


def run_svm_experiment(base_dir="archive/Summarized", text_col="summarized", output_dir="results/svm"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load stratified data
    train_df, val_df, test_df = get_stratified_splits(base_dir=base_dir, text_col=text_col)
    
    # 2. Initialize and train model
    model = SVMTextClassifier(max_features=10000, ngram_range=(1, 2), C=1.0)
    model.fit(train_df['text'].tolist(), train_df['label'].tolist())
    
    # 3. Save model
    model_save_path = "saved_models/svm_tfidf_pipeline.joblib"
    model.save(model_save_path)
    
    # 4. Evaluate on Validation Set
    val_preds = model.predict(val_df['text'].tolist())
    val_probs = model.predict_proba(val_df['text'].tolist())
    val_metrics = evaluate_predictions(val_df['label'].tolist(), val_preds, val_probs, model_name="SVM (Validation)")
    print_metrics_summary(val_metrics)
    save_metrics_to_json(val_metrics, os.path.join(output_dir, "val_metrics.json"))
    
    # 5. Evaluate on Test Set (Final reporting)
    test_preds = model.predict(test_df['text'].tolist())
    test_probs = model.predict_proba(test_df['text'].tolist())
    test_metrics = evaluate_predictions(test_df['label'].tolist(), test_preds, test_probs, model_name="SVM + TF-IDF (Test)")
    print_metrics_summary(test_metrics)
    
    save_metrics_to_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))
    plot_and_save_confusion_matrix(
        test_metrics['confusion_matrix'],
        labels=["Non-Hoax", "Hoax"],
        title="SVM + TF-IDF Confusion Matrix",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM with TF-IDF")
    parser.add_argument("--base_dir", type=str, default="archive/Summarized")
    parser.add_argument("--text_col", type=str, default="summarized")
    parser.add_argument("--output_dir", type=str, default="results/svm")
    args = parser.parse_args()
    
    run_svm_experiment(base_dir=args.base_dir, text_col=args.text_col, output_dir=args.output_dir)
