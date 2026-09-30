import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(base_dir="archive/Summarized", text_col="summarized"):
    """
    Loads and merges the 4 political news datasets (CNN, Kompas, Tempo, TurnBackHoax).
    
    Args:
        base_dir: Path to folder containing the xlsx files (Summarized or Cleaned).
        text_col: Column name to use as feature text ('summarized', 'cleaned', or 'text_new').
        
    Returns:
        pd.DataFrame with columns ['text', 'label', 'source']
    """
    sources_config = [
        {"file": "dataset_cnn_summarized.xlsx", "label": 0, "source": "CNN"},
        {"file": "dataset_kompas_summarized.xlsx", "label": 0, "source": "Kompas"},
        {"file": "dataset_tempo_summarized.xlsx", "label": 0, "source": "Tempo"},
        {"file": "dataset_turnbackhoax_summarized.xlsx", "label": 1, "source": "TurnBackHoax"}
    ]
    
    # Check if base_dir is Cleaned folder
    if "cleaned" in base_dir.lower():
        sources_config = [
            {"file": "dataset_cnn_10k_cleaned.xlsx" if "archive" in base_dir else "dataset_cnn_cleaned.xlsx", "label": 0, "source": "CNN", "col": "text_new"},
            {"file": "dataset_kompas_4k_cleaned.xlsx" if "archive" in base_dir else "dataset_kompas_cleaned.xlsx", "label": 0, "source": "Kompas", "col": "text_new"},
            {"file": "dataset_tempo_6k_cleaned.xlsx" if "archive" in base_dir else "dataset_tempo_cleaned.xlsx", "label": 0, "source": "Tempo", "col": "text_new"},
            {"file": "dataset_turnbackhoax_10_cleaned.xlsx" if "archive" in base_dir else "dataset_turnbackhoax_cleaned.xlsx", "label": 1, "source": "TurnBackHoax", "col": "Clean Narasi"}
        ]
        
    dfs = []
    for item in sources_config:
        file_path = os.path.join(base_dir, item["file"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_excel(file_path)
        active_text_col = item.get("col", text_col)
        
        if active_text_col not in df.columns:
            # Fallback to available text columns
            candidates = [c for c in ['summarized', 'cleaned', 'text_new', 'Clean Narasi', 'FullText'] if c in df.columns]
            if candidates:
                active_text_col = candidates[0]
            else:
                raise KeyError(f"None of candidate text columns found in {item['file']}. Cols: {df.columns}")
                
        label_col = 'label' if 'label' in df.columns else 'hoax'
        
        subset = pd.DataFrame({
            'text': df[active_text_col].astype(str),
            'label': df[label_col].astype(int),
            'source': item["source"]
        })
        dfs.append(subset)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    # Filter empty or null texts
    combined_df = combined_df[combined_df['text'].str.strip().str.len() > 0].reset_index(drop=True)
    
    print(f"[DataLoader] Total loaded samples: {len(combined_df)}")
    print(f"[DataLoader] Class distribution: {dict(combined_df['label'].value_counts())}")
    print(f"[DataLoader] Source breakdown: {dict(combined_df['source'].value_counts())}")
    
    return combined_df


def get_stratified_splits(base_dir="archive/Summarized", 
                           text_col="summarized", 
                           train_ratio=0.8, 
                           val_ratio=0.1, 
                           test_ratio=0.1, 
                           random_state=42, 
                           cache_dir="data_splits",
                           force_recreate=True):
    """
    Creates or loads stratified train, validation, and test splits.
    Ensures identical splits across all 3 models for journal reproducibility.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset_tag = "archive" if "archive" in base_dir else "politics"
    train_path = os.path.join(cache_dir, f"train_{dataset_tag}_{text_col}.csv")
    val_path = os.path.join(cache_dir, f"val_{dataset_tag}_{text_col}.csv")
    test_path = os.path.join(cache_dir, f"test_{dataset_tag}_{text_col}.csv")
    
    if not force_recreate and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"[DataLoader] Loading cached stratified splits from '{cache_dir}'...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df
        
    print("[DataLoader] Generating fresh stratified Train/Val/Test splits...")
    df = load_dataset(base_dir=base_dir, text_col=text_col)
    
    # First split: Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df, 
        test_size=val_test_ratio, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Second split: Val vs Test
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df, 
        test_size=relative_test_ratio, 
        random_state=random_state, 
        stratify=val_test_df['label']
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[DataLoader] Splits saved successfully to '{cache_dir}'.")
    print(f"  Train: {len(train_df)} samples ({dict(train_df['label'].value_counts())})")
    print(f"  Val:   {len(val_df)} samples ({dict(val_df['label'].value_counts())})")
    print(f"  Test:  {len(test_df)} samples ({dict(test_df['label'].value_counts())})")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = get_stratified_splits()
