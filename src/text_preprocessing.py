import re
import collections
import torch


def clean_text_id(text):
    """
    Standard Indonesian text cleaning:
    - Lowercase
    - Remove URLs and HTML tags
    - Normalize repetitive whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    # Remove non-word characters except common Indonesian punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextVocabulary:
    """
    Vocabulary builder for TextCNN with indexing and padding support.
    """
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.pad_idx = 0
        self.unk_idx = 1

    def build_vocab(self, texts, max_vocab_size=30000, min_freq=2):
        counter = collections.Counter()
        for text in texts:
            cleaned = clean_text_id(text)
            tokens = cleaned.split()
            counter.update(tokens)
            
        # Filter by min_freq and take most common up to max_vocab_size
        sorted_words = [word for word, count in counter.most_common() if count >= min_freq]
        if max_vocab_size:
            sorted_words = sorted_words[:max_vocab_size - len(self.word2idx)]
            
        for word in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        print(f"[Vocab] Built vocabulary of size {len(self.word2idx)} (min_freq={min_freq})")

    def encode(self, text, max_len=256):
        cleaned = clean_text_id(text)
        tokens = cleaned.split()
        indices = [self.word2idx.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)
